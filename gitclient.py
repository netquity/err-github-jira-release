"""
This module provides high-level support for managing local git repositories and consolidating changes with their
respective GitHub remotes.
"""

import logging
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from inspect import getfullargspec
from subprocess import CompletedProcess, CalledProcessError

from typing import Callable, List, Generator, Optional

from github import Github
from github.Repository import Repository
from github.Tag import Tag
from github.PaginatedList import PaginatedList

import helpers

logger = logging.getLogger(__file__)

DOMAIN = 'https://github.com'


class GitCommandError(Exception):
    """A git command has failed"""


def _execute_path_git(project_root: str, git_command: list) -> CompletedProcess:
    """Execute a git command in a specific directory

    Conventions:
        - `origin` is used to refer to the remote repo that a project was originally cloned from; there is only one per
          repo
        - `repo` is used to refer to the local repo that was cloned from origin
    """
    try:
        return helpers.run_subprocess(
            ['git'] + git_command,
            cwd=project_root,
        )
    except CalledProcessError:
        logger.exception(
            '%s: Failed git command=%s, output=%s',
            project_root,
            git_command,
            sys.exc_info()[1].stdout,
        )
        raise GitCommandError()


class GitClient:
    """Manage local repos and their remotes"""

    class TagData:
        """A simple wrapper around `github.Tag.Tag` to provide just the details we need"""
        def __init__(self, project_name: str, tag: Tag):
            if not isinstance(tag, Tag):
                raise ValueError(
                    f'Inappropriate type: `tag` argument must be of type `github.Tag.Tag` but got `{type(tag)}`'
                )
            self._project_name = project_name
            self._tag = tag

        @property
        def project_name(self):
            """Get the name of the project this tag belongs to"""
            return self._project_name

        @property
        def name(self):
            """Get the name of the tag: e.g. v5.0.0"""
            return self._tag.name

        @property
        def sha(self):
            """Get the short hash of the commit the tag is pointing at"""
            return self._tag.commit.sha[:7]

        @property
        def url(self):
            """Get the URL of the GitHub release that corresponds with the tag"""
            return f'{DOMAIN}/{self.project_name}/releases/tag/{self.name}'

        @property
        def date(self):
            """Get the last modified date of the tag"""
            return self._tag.commit.stats.last_modified

        @staticmethod
        def is_final_name(tag_name: str) -> bool:
            """Determine whether the given tag string is a final tag string

            >>> GitClient.TagData.is_final_name('v1.0.0')
            True
            >>> GitClient.TagData.is_final_name('v1.0.0-rc.1')
            False
            >>> GitClient.TagData.is_final_name('v1.0.0-rc.1+sealed')
            False
            >>> GitClient.TagData.is_final_name('v1.0.0+20130313144700')
            True
            """
            import semver
            try:
                # tag_name[1:] because our tags have a leading `v`
                return semver.parse(tag_name[1:]).get('prerelease') is None
            except ValueError as exc:
                logger.exception(
                    'Failure parsing tag string=%s',
                    tag_name,
                )
                raise exc

        @staticmethod
        def is_older_name(old_tag_name: str, new_tag_name: str) -> bool:
            """Compare two version strings to determine if one is newer than the other

            :param old_version: version string expected to be sorted before the new_version
            :param new_version: version string expected to be sorted after the old_string
            :return: True if expectations are correct and False otherwise
            >>> GitClient.TagData.is_older_name('v1.0.0', 'v2.0.0')
            True
            >>> GitClient.TagData.is_older_name('v1.0.0', 'v1.0.0')
            False
            >>> GitClient.TagData.is_older_name('v1.0.0', 'v1.0.0-rc.1')
            False
            >>> GitClient.TagData.is_older_name('v1.0.0', 'v1.0.1-rc.1+sealed')
            True
            >>> GitClient.TagData.is_older_name('1.0.0', '2.0.0')  # need to include the leading `v`
            Traceback (most recent call last):
                ...
            ValueError: .0.0 is not valid SemVer string
            """
            from semver import match
            return match(old_tag_name[1:], f"<{new_tag_name[1:]}")

        @staticmethod
        def get_compare_url(project_name: str, old_tag_name: str, new_tag_name: str) -> str:
            """Get the URL to compare two tags on GitHub"""
            return f'{DOMAIN}/{project_name}/compare/{old_tag_name}...{new_tag_name}'

    def __init__(self, config: dict):
        self.repos_root = config['REPOS_ROOT']
        self.project_names = config['PROJECT_NAMES']
        self.github = Github(config['GITHUB_TOKEN'])

    @contextmanager
    def _gcmd(self, project_name: str) -> Generator[CompletedProcess, None, None]:
        """A context manager for interacting with local git repos in a safer way

        Records reflog position before doing anything and resets to this position if any of the git commands fail.
        """
        initial_ref = self.get_latest_ref(project_name)
        backup_path = self._backup_repo(project_name)
        try:
            yield lambda cmd: self._execute_project_git(project_name, cmd)
        except GitCommandError as exc:
            self._restore_repo(project_name, backup_path)
            logger.error('%s: git commands failed; repo backup %s restored', project_name, initial_ref)
            raise exc

    @contextmanager
    def project_git(self, project_name: str):
        """Context wrapper to provide the project_name in a more limited scope"""
        class ProjectGit:  # pylint:disable=too-few-public-methods
            """A proxy object for marshalling calls to the underlying gitclient"""
            def __init__(self, project_name: str, git: GitClient):
                self._project_name = project_name
                self._git = git

            def __getattribute__(self, name: str):
                """Intercept calls to gitclient methods and project_name argument if the target method neets it"""
                if name in ['_git', '_project_name']:
                    return super(ProjectGit, self).__getattribute__(name)
                method = self._git.__getattribute__(name)
                if 'project_name' not in getfullargspec(method).args:
                    return method
                return partial(method, project_name=self._project_name)

        yield ProjectGit(project_name, self)

    def get_latest_ref(self, project_name: str) -> str:
        """Get the latest rev hash from reflog"""
        return self._execute_project_git(
            project_name,
            ['reflog', 'show', '--format=%H', '-1']
        ).stdout.splitlines()[0]

    def clean(self, project_name: str) -> CompletedProcess:
        """Recursively remove files that aren't under source control"""
        return self._execute_project_git(self._get_project_root(project_name), ['clean', '-f'])

    def reset_hard(self, project_name: str, ref: str) -> CompletedProcess:
        """Do a hard reset on a repo to a target ref"""
        return self._execute_project_git(self._get_project_root(project_name), ['reset', '--hard', ref])

    def create_tag(self, project_name: str, tag_name: str) -> None:  # TODO: return TagData
        """Create a git tag on whatever commit HEAD is pointing at"""
        tag_name = f'v{tag_name}'
        self._execute_project_git(
            project_name,
            ['tag', '-s', tag_name, '-m', tag_name, ]
        )

    def create_ref(self, project_name: str, new_version_name: str, ref: str = 'master') -> None:
        """Create a ref and push it to origin

        https://developer.github.com/v3/git/refs/#create-a-reference
        """
        self._get_origin(project_name).create_git_ref(
            f'refs/tags/v{new_version_name}',
            self.get_rev_hash(project_name, ref),  # TODO: this will have to be something else for hotfixes
        )

    def create_release(self, project_name: str, release_notes: str, new_version_name: str):
        """Create a GitHub release object and push it origin

        https://developer.github.com/v3/repos/releases/#create-a-release
        """
        self._get_origin(project_name).create_git_release(
            tag=f'v{new_version_name}',
            name=f'{project_name} - Version {new_version_name}',
            message=release_notes,
            draft=False,
            prerelease=False,
        )

    def tag_develop(self, project_name: str, tag_name: str) -> None:
        """Compound check, tag, ref creation and pushing it all to origin

        :param project_name:
        :param tag_name: the new tag to apply to develop's HEAD
        """
        self.checkout_latest(project_name, 'develop')
        self.create_tag(project_name, tag_name)
        self.create_ref(project_name, tag_name, 'develop')

    def get_rev_hash(self, project_name: str, ref: str) -> str:
        """Get the SHA1 hash of a particular git ref"""
        return self._execute_project_git(
            project_name,
            ['rev-parse', ref]
        ).stdout.strip()  # Get rid of the newline character at the end

    def merge_and_create_release_commit(
            self, project_name: str, new_version_name: str,
            release_notes: str, changelog_path: str
    ) -> None:
        """Create a release commit based on origin/develop and merge it to master"""
        with self._gcmd(project_name) as gcmd:
            for git_command in [
                    # TODO: deal with merge conflicts in an interactive way
                    ['fetch', '-p'],
                    ['checkout', '-B', f'release-{new_version_name}', 'origin/develop'],
            ]:
                gcmd(git_command)

            helpers.update_changelog_file(
                changelog_path,
                release_notes,
                logger,
            )

            for git_command in [
                    ['add', changelog_path],
                    ['commit', '-m', f'Release {new_version_name}'],
                    ['checkout', '-B', 'master', 'origin/master'],
                    ['merge', '--no-ff', '--no-edit', 'release-{new_version_name}'],
                    ['push', 'origin', 'master'],
            ]:
                gcmd(git_command)

    def update_develop(self, project_name: str) -> None:
        """Merge master branch to develop"""
        with self._gcmd(project_name) as gcmd:
            for git_command in [
                    ['fetch', '-p'],
                    ['checkout', '-B', 'develop', 'origin/develop'],
                    ['merge', '--no-ff', '--no-edit', 'origin/master'],
                    ['push', 'origin', 'develop'],
            ]:
                gcmd(git_command)

    def checkout_latest(self, project_name: str, ref: str) -> None:
        """Check out the latest version of develop for the given project"""
        with self._gcmd(project_name) as gcmd:
            for git_command in [
                    ['fetch', '--prune', '--force', '--tags'],
                    ['checkout', '-B', ref, f'origin/{ref}'],
            ]:
                gcmd(git_command)

    def add_release_notes_to_develop(self, project_name: str, new_version_name: str, release_notes: str) -> str:
        """Wrap subprocess calls with some project-specific defaults.

        :return: Release commit hash.
        """
        with self._gcmd(project_name) as gcmd:
            for git_command in [
                    # TODO: deal with merge conflicts in an interactive way
                    ['fetch', '-p'],
                    ['checkout', '-B', 'develop', 'origin/develop'],
            ]:
                gcmd(git_command)

                changelog_path = os.path.join(
                    self._get_project_root(project_name), 'CHANGELOG.md',
                )
                helpers.update_changelog_file(changelog_path, release_notes, logger)

            for git_command in [
                    ['add', changelog_path],
                    # FIXME: make sure version number doesn't have `hotfix` in it, or does... just make it consistent
                    ['commit', '-m', f'Hotfix {new_version_name}'],
                    ['push', 'origin', 'develop'],
            ]:
                gcmd(git_command)

        return self.get_rev_hash(project_name, 'develop')

    def get_updated_repo_names(self, project_names: List[str], since_final: bool = True) -> List[str]:
        """
        Get a list of the full names of the repos that have commits to develop since either the last final or
        pre-release

        Note that when the develop branch is updated immediately after a release, it creates a merge commit, which is
        not counted for the purposes of this method.

        :param since_final: if True, look for updates since the latest final tag; otherwise, since latest pre-release
        """
        return [project.full_name for project in self._get_updated_origins(project_names, since_final)]

    def get_merge_count(self, project_name: str) -> int:
        """Get the number of merges to develop since the last final tag"""
        return self._get_merge_count_since(
            project_name,
            GitClient._get_latest_tag(
                origin=self._get_origin(project_name),
                find_final=True,
            ),
        )

    def get_latest_pre_release_tag_name(self, project_name: str, min_version: str = None) -> Optional[str]:
        """Get the latest pre-release tag name

        :param min_version: if included, will ignore all versions below this one
        :return: either the version string of the latest pre-release tag or `None` if one wasn't found
        """
        try:
            latest_pre_tag = GitClient._get_latest_tag(self._get_origin(project_name), False).name
            if not min_version:
                return latest_pre_tag
        except AttributeError:
            return None
        return latest_pre_tag if GitClient.TagData.is_older_name(min_version, latest_pre_tag) else None

    def get_latest_compare_url(self, project_name: str) -> str:
        """Get the URL to compare the latest final with the latest pre-release on GitHub"""
        latest_final = self.get_latest_final_tag(project_name).name
        return self.TagData.get_compare_url(
            project_name,
            old_tag_name=latest_final,
            new_tag_name=self.get_latest_pre_release_tag_name(
                project_name,
                min_version=latest_final,
            )
        )

    def get_latest_merged_prs_url(self, project_name: str) -> str:
        """Get the URL to see merged PRs since the latest final on GitHub"""
        start_date = GitClient._parse_github_datetime(self.get_latest_final_tag(project_name).date)
        return GitClient._get_merged_prs_url(
            project_name,
            start_date.isoformat(),  # TODO: timezone?
            datetime.now(timezone.utc).replace(microsecond=0).isoformat().split('+')[0],
        )

    # TODO: this is very Django specific, figure out less opinionated way for non-Django users
    def get_migration_count(self, project_name: str) -> int:
        """Get the number of new migration files since the latest final"""
        tag_name = self.get_latest_final_tag(project_name).name
        return len(self._execute_project_git(
            project_name,
            [
                'diff', '--name-status', '--diff-filter=A',
                f'HEAD..{tag_name}', '--', 'src/**/migrations/',
            ]
        ).stdout.strip().splitlines())

    def get_latest_final_tag(self, project_name: str) -> Optional['TagData']:
        """Get the latest final tag for a given project name"""
        return GitClient._get_latest_tag(
            origin=self._get_origin(project_name),
            find_final=True,
        )

    def _get_origins(self, project_names: List[str]) -> List[Repository]:
        """Get a list of the `origin` repos for each of the given project names

        http://developer.github.com/v3/repos/
        """
        return [self._get_origin(project_name) for project_name in project_names]

    def _get_origin(self, project_name: str):
        """Get the remote repo that a project was cloned from

        Note: only a single remote, `origin`, is currently supported.
        """
        return self.github.get_repo(project_name)

    def _is_updated_since(self, origin: Repository, since_final: bool = True) -> bool:
        """Check if the given origin has commits to develop since either the last final or pre-release"""
        return self._get_merge_count_since(
            origin.full_name,
            GitClient._get_latest_tag(origin, since_final)
        ) > 0

    def _get_updated_origins(self, project_names: List[str], since_final: bool = True) -> List[Repository]:
        """Get a list of the `origin` repos that have commits to develop since either the last final or pre-release

        :param since_final: if True, look for updates since the latest final tag; otherwise, since latest pre-release
        """
        return [
            origin for origin in self._get_origins(project_names)
            if self._is_updated_since(origin, since_final)
        ]

    def _get_merge_count_since(self, project_name: str, tag: TagData) -> int:
        """Get the number of merges to develop since the given tag"""
        # FIXME: assumes master and developed have not diverged, which is not a safe assumption at all
        return len(self._execute_project_git(
            project_name,
            ['log', f'{tag.name}...origin/develop', '--merges', '--oneline', ]
        ).stdout.splitlines()) - 1  # The first result will be the merge commit from last release

    def _backup_repo(self, project_name: str) -> str:
        """Create a backup of the entire local repo folder and return the destination

        :return: the dst path
        """
        # TODO: maybe it would be better to back up the whole repos root, instead of individual repos
        ref = self.get_latest_ref(project_name)[:7]
        return helpers.copytree(
            self._get_project_root(project_name),
            self._get_backups_path(project_name),
            ref,
        )

    def _restore_repo(self, project_name: str, backup_path: str) -> str:
        """Restore a repo backup directory to its original location

        :param backup_path: absolute path to the backup's root as returned by `_backup_repo()`
        """
        # create a backup of the backup so it can be moved using the atomic `shutil.move`
        backup_swap = helpers.copytree(
            src=backup_path,
            dst_parent=self._get_backups_path(project_name),
            dst=self.get_latest_ref(project_name)[:7] + '.swap',
        )
        # move the backup to the normal repo location
        project_root = self._get_project_root(project_name)
        shutil.rmtree(project_root)
        return shutil.move(src=backup_swap, dst=project_root)

    def _get_backups_path(self, project_name: str = None) -> str:
        """Get the backups dir path, either for all projects, or for the given project name"""
        return os.path.join(
            *[self.repos_root, '.backups']
            + ([project_name] if project_name else [])
        )

    def _execute_project_git(self, project_name: str, git_command: list) -> CompletedProcess:
        """Simple wrapper for executing git commands by project name"""
        return _execute_path_git(self._get_project_root(project_name), git_command)

    def _get_project_root(self, project_name: str) -> str:
        """Get the full path to the project root"""
        return os.path.join(self.repos_root, project_name)

    @classmethod
    def _get_latest_tag(cls, origin: Repository, find_final: bool = True) -> Optional['TagData']:
        """Get the latest final or pre-release tag

        Final tags do not contain a pre-release segment, but may contain a SemVer metadata segment.

        Tags are identified as pre-release tags if they contain a pre-release segment such as the following, where the
        hyphen-separated component (`rc.1`) makes up the pre-release segment:
        v1.0.0-rc.1
        v1.0.0-rc.1+sealed

        However, the presence of a SemVer metadata segment has no bearing on whether it's a pre-release tag or not.

        :param find_final: if True, look for the latest final tag; otherwise, look for latest pre-release
        """
        return cls._find_tag(
            origin,
            cls.TagData.is_final_name if find_final
            else lambda tag: not cls.TagData.is_final_name(tag)
        )

    @classmethod
    def _find_tag(cls, origin: Repository, test: Callable[[str], bool]) -> Optional['TagData']:
        """Return the first tag that passes a given test or `None` if none found"""
        try:
            return cls.TagData(
                origin.full_name,
                next((tag for tag in cls._get_tags(origin) if test(tag.name)), None),
            )
        except ValueError:
            return None

    @staticmethod
    def _get_tags(origin: Repository) -> PaginatedList:
        """Get all the tags for the repo

        :return: `github.PaginatedList.PaginatedList` of `github.Tag.Tag`
        """
        return origin.get_tags()  # TODO: consider searching local repo instead of GitHub

    @staticmethod
    def _parse_github_datetime(dt: str) -> datetime:  # pylint:disable=invalid-name
        """Take GitHub's ISO8601 datetime string and return a datetime object

        >>> GitClient._parse_github_datetime('Thu, 28 Feb 2019 17:24:21 GMT')
        datetime.datetime(2019, 2, 28, 17, 24, 21)
        """
        return datetime.strptime(dt, '%a, %d %b %Y %H:%M:%S %Z')

    @staticmethod
    def _get_merged_prs_url(project_name: str, start_date: str, end_date: str) -> str:
        """Get the URL to see merged PRs in a date range on GitHub

        >>> GitClient._get_merged_prs_url('foo/bar-prj', '2018-01-01T22:02:39+00:00', '2018-01-02T22:02:39+00:00')[:46]
        'https://github.com/foo/bar-prj/pulls?utf8=✓&q='
        >>> GitClient._get_merged_prs_url('foo/bar-prj', '2018-01-01T22:02:39+00:00', '2018-01-02T22:02:39+00:00')[46:]
        'is:pr+is:closed+merged:2018-01-01T22:02:39+00:00..2018-01-02T22:02:39+00:00'
        """
        return f'{DOMAIN}/{project_name}/pulls?utf8=✓&q=is:pr+is:closed+merged:{start_date}..{end_date}'
