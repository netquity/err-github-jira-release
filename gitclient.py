"""
This module provides high-level support for managing local git repositories and consolidating changes with their
respective GitHub remotes.
"""

import logging
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone

from typing import Union, Callable, List, Generator

from github import Github, PaginatedList
from github.Repository import Repository

import helpers

logger = logging.getLogger(__file__)


class GitCommandError(Exception):
    """A git command has failed"""


def _execute_path_git(project_root: str, git_command: list) -> subprocess.CompletedProcess:
    """Execute a git command in a specific directory"""
    try:
        return helpers.run_subprocess(
            ['git'] + git_command,
            cwd=project_root,
        )
    except subprocess.CalledProcessError:
        logger.exception(
            '%s: Failed git command=%s, output=%s',
            project_root,
            git_command,
            sys.exc_info()[1].stdout,
        )
        raise GitCommandError()


class GitClient:
    """Manage local repos and their remotes"""
    def __init__(self, config: dict):
        self.repos_root = config['REPOS_ROOT']
        self.project_names = config['PROJECT_NAMES']
        self.github = Github(config['GITHUB_TOKEN'])

    @contextmanager
    def _gcmd(self, project_name: str) -> Generator[subprocess.CompletedProcess, None, None]:
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

    def get_latest_ref(self, project_name: str) -> str:
        """Get the latest rev hash from reflog"""
        return self._execute_project_git(
            project_name,
            ['reflog', 'show', '--format=%H', '-1']
        ).stdout.splitlines()[0]

    def clean(self, project_name: str) -> subprocess.CompletedProcess:
        """Recursively remove files that aren't under source control"""
        return self._execute_project_git(self._get_project_root(project_name), ['clean', '-f'])

    def reset_hard(self, project_name: str, ref: str) -> subprocess.CompletedProcess:
        return self._execute_project_git(self._get_project_root(project_name), ['reset', '--hard', ref])

    def create_tag(self, project_name: str, tag_name: str) -> None:
        """Create a git tag on whatever commit HEAD is pointing at"""
        tag_name = f'v{tag_name}'
        self._execute_project_git(
            project_name,
            ['tag', '-s', tag_name, '-m', tag_name, ]
        )

    def create_ref(self, project_name: str, new_version_name: str) -> None:
        """Create a ref and push it to origin

        https://developer.github.com/v3/git/refs/#create-a-reference
        """
        self._get_remote_repo(project_name).create_git_ref(
            f'refs/tags/v{new_version_name}',
            self.get_rev_hash(project_name, 'master'),  # TODO: this will have to be something else for hotfixes
        )

    def create_release(self, project_name: str, release_notes: str, new_version_name: str):
        """Create a GitHub release object and push it origin

        https://developer.github.com/v3/repos/releases/#create-a-release
        """
        self._get_remote_repo(project_name).create_git_release(
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
        self.create_ref(project_name, tag_name)

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

    def get_all_repos(self, project_names: List[str]) -> List[Repository]:
        """Get a list of all the repositories under management

        http://developer.github.com/v3/repos/
        """
        return [self._get_remote_repo(project_name) for project_name in project_names]

    def get_updated_repo_names(self, project_names: List[str]) -> List[str]:
        """Get a list of the full names of the repos that have commits to develop since last final release"""
        return [project.full_name for project in self._get_updated_repos(project_names)]

    def get_merge_count(self, project_name: str) -> int:
        """Get the number of merges to develop since the last final tag"""
        return self._get_merge_count_since(
            project_name,
            self._get_latest_final_tag(
                self._get_remote_repo(project_name),
            ).name,
        )

    def get_latest_pre_release_tag_name(self, project_name: str, min_version: str = None) -> Union[str, None]:
        """Get the latest pre-release tag name

        :param min_version: if included, will ignore all versions below this one
        :return: either the version string of the latest pre-release tag or `None` if one wasn't found
        """
        latest_pre_tag = GitClient._get_latest_pre_release_tag(self._get_remote_repo(project_name)).name
        if not min_version:
            return latest_pre_tag
        if GitClient._is_older_version(min_version, latest_pre_tag):
            return latest_pre_tag
        return None

    def get_latest_final_tag_name(self, project_name: str) -> str:
        """Get the latest final release's tag name"""
        return GitClient._get_latest_final_tag(self._get_remote_repo(project_name)).name

    def get_latest_final_tag_sha(self, project_name: str) -> str:
        """Get the latest final release's tag sha"""
        return GitClient._get_latest_final_tag(self._get_remote_repo(project_name)).commit.sha

    def get_latest_final_tag_url(self, project_name: str) -> str:
        """Get the latest final release's tag GitHub URL"""
        return GitClient.release_url(
            project_name,
            self.get_latest_final_tag_name(project_name),
        )

    def get_latest_final_tag_date(self, project_name: str) -> str:
        """Get the latest final release's tag date"""
        return GitClient._get_latest_final_tag(self._get_remote_repo(project_name)).commit.stats.last_modified

    def get_latest_compare_url(self, project_name: str) -> str:
        """Get the URL to compare the latest final with the latest pre-release on GitHub"""
        latest_final = self.get_latest_final_tag_name(project_name)
        return self._get_compare_url(
            project_name,
            old_tag=latest_final,
            new_tag=self.get_latest_pre_release_tag_name(
                project_name,
                min_version=latest_final,
            )
        )

    def get_latest_merged_prs_url(self, project_name: str) -> str:
        """Get the URL to see merged PRs since the latest final on GitHub"""
        start_date = GitClient._parse_github_datetime(self.get_latest_final_tag_date(project_name))
        return GitClient._get_merged_prs_url(
            project_name,
            start_date.isoformat(),  # TODO: timezone?
            datetime.now(timezone.utc).replace(microsecond=0).isoformat().split('+')[0],
        )

    # TODO: this is very Django specific, figure out less opinionated way for non-Django users
    def get_migration_count(self, project_name: str) -> int:
        tag_name = self.get_latest_final_tag_name(project_name)
        return len(self._execute_project_git(
            project_name,
            [
                'diff', '--name-status', '--diff-filter=A',
                f'HEAD..{tag_name}', '--', 'src/**/migrations/',
            ]
        ).stdout.strip().splitlines())

    def _get_remote_repo(self, project_name: str):
        return self.github.get_repo(project_name)

    def _is_updated_since_last_final(self, repo: Repository) -> bool:
        """Check if the given repo has commits to develop since the last final release"""
        return self._get_merge_count_since(
            repo.full_name,
            GitClient._get_latest_final_tag(repo).name
        ) > 0

    def _get_updated_repos(self, project_names: List[str]) -> List[Repository]:
        """Get a list of repos that have commits to develop since the last final release"""
        return [
            repo for repo in self.get_all_repos(project_names)
            if self._is_updated_since_last_final(repo)
        ]

    def _get_merge_count_since(self, project_name: str, tag_name: str) -> int:
        """Get the number of merges to develop since the given tag"""
        # FIXME: assumes master and developed have not diverged, which is not a safe assumption at all
        return len(self._execute_project_git(
            project_name,
            ['log', f'{tag_name}...origin/develop', '--merges', '--oneline', ]
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

    def _execute_project_git(self, project_name: str, git_command: list) -> subprocess.CompletedProcess:
        """Simple wrapper for executing git commands by project name"""
        return _execute_path_git(self._get_project_root(project_name), git_command)

    def _get_project_root(self, project_name: str) -> str:
        """Get the full path to the project root"""
        return os.path.join(self.repos_root, project_name)

    @classmethod
    def _get_latest_pre_release_tag(cls, origin: Repository) -> Union['github.Tag.tag', None]:
        """Get the latest pre-release tag

        Tags are identified as pre-release tags if they contain a pre-release segment such as the following, where the
        hyphen-separated component (`rc.1`) makes up the pre-release segment:
        v1.0.0-rc.1
        v1.0.0-rc.1+sealed

        However, the presence of a SemVer metadata segment has no bearing on whether it's a pre-release tag or not.
        """
        return cls._find_tag(origin, cls._is_prerelease_tag_name)

    @classmethod
    def _get_latest_final_tag(cls, origin: Repository) -> Union['github.Tag.tag', None]:
        """Get the latest final tag

        Final tags do not contain a pre-release segment, but may contain a SemVer metadata segment.
        """
        return cls._find_tag(origin, lambda tag: not cls._is_prerelease_tag_name(tag))

    @classmethod
    def _find_tag(cls, origin: Repository, test: Callable[[str], bool]) -> Union['github.Tag.tag', None]:
        """Return the first tag that passes a given test or `None` if none found"""
        return next((tag for tag in cls._get_tags(origin) if test(tag.name)), None)

    @staticmethod
    def _get_tags(origin: Repository) -> PaginatedList.PaginatedList:
        """Get all the tags for the repo"""
        return origin.get_tags()  # TODO: consider searching local repo instead of GitHub

    @staticmethod
    def _is_older_version(old_version: str, new_version: str) -> bool:
        """Compare two version strings to determine if one is newer than the other

        :param old_version: version string expected to be sorted before the new_version
        :param new_version: version string expected to be sorted after the old_string
        :return: True if expectations are correct and False otherwise
        >>> GitClient._is_older_version('v1.0.0', 'v2.0.0')
        True
        >>> GitClient._is_older_version('v1.0.0', 'v1.0.0')
        False
        >>> GitClient._is_older_version('v1.0.0', 'v1.0.0-rc.1')
        False
        >>> GitClient._is_older_version('v1.0.0', 'v1.0.1-rc.1+sealed')
        True
        >>> GitClient._is_older_version('1.0.0', '2.0.0')  # need to include the leading `v`
        Traceback (most recent call last):
            ...
        ValueError: .0.0 is not valid SemVer string
        """
        from semver import match
        return match(old_version[1:], f"<{new_version[1:]}")

    @staticmethod
    def _get_compare_url(project_name: str, old_tag: str, new_tag: str) -> str:
        """Get the URL to compare two tags on GitHub"""
        return f'https://github.com/{project_name}/compare/{old_tag}...{new_tag}'

    @staticmethod
    def _parse_github_datetime(dt: str) -> datetime:
        """Take GitHub's ISO8601 datetime string and return a datetime object

        >>> GitClient._parse_github_datetime('Thu, 28 Feb 2019 17:24:21 GMT')
        datetime.datetime(2019, 2, 28, 17, 24, 21)
        """
        return datetime.strptime(dt, '%a, %d %b %Y %H:%M:%S %Z')

    @staticmethod
    def _get_merged_prs_url(project_name: str, start_date: str, end_date: str) -> str:
        """Get the URL to see merged PRs in a date range on GitHub

        >>> GitClient._get_merged_prs_url('foo/bar-project', '2018-01-01T22:02:39+00:00', '2018-01-02T22:02:39+00:00')
        'https://github.com/foo/bar-project/pulls?utf8=✓&q=is:pr+is:closed+merged:2018-01-01T22:02:39+00:00..2018-01-02T22:02:39+00:00'
        """
        return f'https://github.com/{project_name}/pulls?utf8=✓&q=is:pr+is:closed+merged:{start_date}..{end_date}'

    @staticmethod
    def _is_prerelease_tag_name(tag_name: str) -> bool:
        """Determine whether the given tag string is a pre-release tag string

        >>> GitClient._is_prerelease_tag_name('v1.0.0')
        False
        >>> GitClient._is_prerelease_tag_name('v1.0.0-rc.1')
        True
        >>> GitClient._is_prerelease_tag_name('v1.0.0-rc.1+sealed')
        True
        >>> GitClient._is_prerelease_tag_name('v1.0.0+20130313144700')
        False
        """
        import semver
        try:
            # tag_name[1:] because our tags have a leading `v`
            return semver.parse(tag_name[1:]).get('prerelease') is not None
        except ValueError as exc:
            logger.exception(
                'Failure parsing tag string=%s',
                tag_name,
            )
            raise exc

    @staticmethod
    def release_url(project_name: str, new_version_name: str) -> str:
        """Get the GitHub release URL"""
        return f'https://github.com/{project_name}/releases/tag/v{new_version_name}'
