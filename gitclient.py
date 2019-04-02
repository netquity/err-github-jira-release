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
from functools import partial, lru_cache
from inspect import getfullargspec
from subprocess import CompletedProcess, CalledProcessError
from collections import namedtuple

from typing import Callable, List, Generator, Optional, Dict, NamedTuple

from github import Github
from github.Repository import Repository
from github.Tag import Tag
from github.PaginatedList import PaginatedList

from packaging.version import parse, InvalidVersion

import helpers

logger = logging.getLogger(__file__)
MergeLog = namedtuple('MergeLog', ['key', 'sha'])  # jira ticket key and commit sha

DOMAIN = 'https://github.com'

TagTup = namedtuple('TagTup', ['sha', 'name', 'date'])  # TODO: rename


class GitCommandError(Exception):
    """A git command has failed"""


@lru_cache(maxsize=500)
def cached_parse(tag_name: str):
    """Parse the tag using `packaging.version`

    Since tag names include the short commit SHA,
    they should be unique enough to cache this way.
    """
    return parse(tag_name)


def format_version(version: str) -> str:
    """Get the given version with a leading `v` character

    >>> format_version('1.0.0')
    'v1.0.0'
    >>> format_version('v1.0.0')
    'v1.0.0'
    >>> format_version('1.0.0+666d074')
    'v1.0.0+666d074'
    >>> format_version('1.0.0+sealed.666d074')
    'v1.0.0+sealed.666d074'
    >>> format_version('v1.0.0+sealed.666d074')
    'v1.0.0+sealed.666d074'
    """
    import re
    return re.sub(r'^(v)?(.*)$', r'v\g<2>', version)


def _execute_path_git(project_root: str, git_command: list) -> CompletedProcess:
    """Execute a git command in a specific directory"""
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
    """Manage local repos and their remotes

    Conventions:
        - `origin` is used to refer to the remote repo that a project was originally cloned from; there is only one per
          repo
        - `repo` is used to refer to the local repo that was cloned from origin
        - Public methods are always dealing with the latest changes. For example, `get_ref` will always get the most
          recent revision hash from the reflog; `get_merged_prs_url` will get show merged PRs since the latest final
          release.
    """
    class ProjectPath(namedtuple('ProjectPath', 'path')):  # TODO: rename, maybe to just `project`
        """Projects have a path and a limited set of attributes that can be derived from it"""
        @property
        def name(self):
            """Get the full name of the project e.g. 'netquity/err-github-jira-release'"""
            import re
            return re.match(r'^.*/(.*/.*)/?$', self.path)[1]  # TODO: pretty slow, make better

        def __repr__(self):
            return f'{self.__class__.__name__}({self.name})'


    class TagData:
        """A simple wrapper around `github.Tag.Tag` to provide just the details we need"""
        __slots__ = ['_project', '_tag']

        def __init__(
                self,
                project: 'GitClient.ProjectPath',
                tag: NamedTuple('TagTup', [
                    ('sha', str),
                    ('name', str),
                    ('date', str)
                ])
        ):
            if not isinstance(tag, TagTup):
                raise TypeError(
                    f'Inappropriate type: `tag` argument must be of type `TagTup` but got `{type(tag)}`'
                )
            self._project = project
            self._tag = tag
            logger.debug('%s inited TagData for %s', project.name, self.name)

        def __repr__(self):
            return f'{self.__class__.__name__}({self.sha}, {self.name})'

        @property
        def project(self):
            """Get the name of the project this tag belongs to"""
            return self._project

        @property
        def name(self):
            """Get the name of the tag: e.g. v5.0.0"""
            return self._tag.name

        @property
        def sha(self):
            """Get the short hash of the commit the tag is pointing at"""
            return self._tag.sha

        @property
        def url(self):
            """Get the URL of the GitHub release that corresponds with the tag"""
            return f'{DOMAIN}/{self.project.name}/releases/tag/{self.name}'

        @property
        def date(self):
            """Get the creatordate of the tag"""
            return self._tag.date

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

            :param old_tag_name: version string expected to be sorted before the new_tag_name
            :param new_tag_name: version string expected to be sorted after the old_tag_name
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

    __slots__ = ['repos_root', 'projects', 'github']

    def __init__(self, config: dict):
        self.repos_root = config['REPOS_ROOT']
        self.projects = [
            GitClient.ProjectPath(os.path.join(self.repos_root, project_name))
            for project_name in config['PROJECT_NAMES']
        ]
        self.github = Github(config['GITHUB_TOKEN'])

    @contextmanager
    def _gcmd(self, project: str) -> Generator[CompletedProcess, None, None]:
        """A context manager for interacting with local git repos in a safer way

        Records reflog position before doing anything and resets to this position if any of the git commands fail.
        """
        initial_ref = self.get_ref(project)
        backup_path = self._backup_repo(project)
        try:
            yield lambda cmd: self._execute_project_git(project, cmd)
        except GitCommandError as exc:
            self._restore_repo(project, backup_path)
            logger.error('%s: git commands failed; repo backup %s restored', project, initial_ref)
            raise exc

    @contextmanager
    def project_git(self, project: str) -> Generator['ProjectGit', None, None]:
        """Context wrapper to provide the project in a more limited scope"""
        class ProjectGit:  # pylint:disable=too-few-public-methods
            """A proxy object for marshalling calls to the underlying GitClient"""
            def __init__(self, project: str, git: GitClient):
                self._project = project
                self._git = git

            def __getattribute__(self, name: str):
                """Intercept calls to GitClient methods and add project argument if the target method neets it"""
                if name in ['_git', '_project']:
                    return super(ProjectGit, self).__getattribute__(name)
                method = self._git.__getattribute__(name)
                if 'project' not in getfullargspec(method).args:
                    return method
                return partial(method, project=self._project)

        yield ProjectGit(project, self)

    def get_ref(self, project: str) -> str:
        """Get the latest rev hash from reflog"""
        ref = self._execute_project_git(
            project,
            ['reflog', 'show', '--format=%H', '-1']
        ).stdout.splitlines()[0]
        logger.debug('%s get latest rev hash from reflog: %s', project, ref)
        return ref

    def clean(self, project: str) -> CompletedProcess:
        """Recursively remove files that aren't under source control"""
        return self._execute_project_git(self._get_project_root(project), ['clean', '-f'])

    def reset_hard(self, project: str, ref: str) -> CompletedProcess:
        """Do a hard reset on a repo to a target ref"""
        return self._execute_project_git(self._get_project_root(project), ['reset', '--hard', ref])

    def create_tag(self, project: str, tag_name: str) -> None:  # TODO: return TagData
        """Create a git tag on whatever commit HEAD is pointing at"""
        tag_name = format_version(tag_name)
        self._execute_project_git(
            project,
            ['tag', '-s', tag_name, '-m', tag_name, ]
        )
        logger.info('%s: created tag %s', project, tag_name)

    def create_ref(self, project: str, version_name: str, ref: str = 'master') -> None:
        """Create a ref and push it to origin

        https://developer.github.com/v3/git/refs/#create-a-reference
        """
        version_name = format_version(version_name)
        self._get_origin(project).create_git_ref(
            f'refs/tags/{version_name}',
            self.get_rev_hash(project, ref),  # TODO: this will have to be something else for hotfixes
        )
        logger.info('%s: created ref %s', project, version_name)

    def create_release(self, project: str, release_notes: str, version_name: str):
        """Create a GitHub release object and push it origin

        https://developer.github.com/v3/repos/releases/#create-a-release
        """
        version_name = format_version(version_name)
        self._get_origin(project).create_git_release(
            tag=version_name,
            name=f'{project} - Version {version_name}',
            message=release_notes,
            draft=False,
            prerelease=False,
        )
        logger.info('%s: created ref %s', project, version_name)

    def tag_develop(self, project: str, tag_name: str) -> None:
        """Compound check, tag, ref creation and pushing it all to origin

        :param project:
        :param tag_name: the new tag to apply to develop's HEAD
        """
        tag_name = format_version(tag_name)
        self.checkout_latest(project, 'develop')
        self.create_tag(project, tag_name)
        self.create_ref(project, tag_name, 'develop')

    def get_rev_hash(self, project: str, ref: str) -> str:
        """Get the SHA1 hash of a particular git ref"""
        rev_hash = self._execute_project_git(
            project,
            ['rev-parse', ref]
        ).stdout.strip()  # Get rid of the newline character at the end
        logger.debug('%s get rev hash of ref %s: %s', project, ref, rev_hash)
        return rev_hash

    def merge_and_create_release_commit(self, project: str, version_name: str, release_notes: str) -> None:
        """Create a release commit based on origin/develop and merge it to master"""
        logger.info('%s start merging and creating release commit for %s', project, version_name)
        with self._gcmd(project) as gcmd:
            for git_command in [
                    # TODO: deal with merge conflicts in an interactive way
                    ['fetch', '-p'],
                    ['checkout', '-B', f'release-{version_name}', 'origin/develop'],
            ]:
                gcmd(git_command)

            changelog_path = os.path.join(
                self._get_project_root(project), 'CHANGELOG.md',
            )
            helpers.update_changelog_file(
                changelog_path,
                release_notes,
                logger,
            )

            for git_command in [
                    ['add', changelog_path],
                    ['commit', '-m', f'Release {version_name}'],
                    ['checkout', '-B', 'master', 'origin/master'],
                    ['merge', '--no-ff', '--no-edit', f'release-{version_name}'],
                    ['push', 'origin', 'master'],
            ]:
                gcmd(git_command)
        logger.info('%s complete merging and creating release commit for %s', project, version_name)

    def update_develop(self, project: str) -> None:
        """Merge master branch to develop"""
        logger.debug('%s start update develop by merging master', project)
        with self._gcmd(project) as gcmd:
            for git_command in [
                    ['fetch', '-p'],
                    ['checkout', '-B', 'develop', 'origin/develop'],
                    ['merge', '--no-ff', '--no-edit', 'origin/master'],
                    ['push', 'origin', 'develop'],
            ]:
                gcmd(git_command)
        logger.debug('%s complete update develop by merging master', project)

    def checkout_latest(self, project: str, ref: str) -> None:
        """Check out the latest version of develop for the given project"""
        with self._gcmd(project) as gcmd:
            for git_command in [
                    ['fetch', '--prune', '--force', '--tags'],
                    ['checkout', '-B', ref, f'origin/{ref}'],
            ]:
                gcmd(git_command)
        logger.info('%s: checked out latest origin/%s', project, ref)

    def add_release_notes_to_develop(self, project: str, version_name: str, release_notes: str) -> str:
        """Wrap subprocess calls with some project-specific defaults.

        :return: Release commit hash.
        """
        logger.info(
            '%s start adding release notes to develop %s (now %s)',
            project,
            version_name,
            self.get_rev_hash(project, 'develop'),
        )
        with self._gcmd(project) as gcmd:
            for git_command in [
                    # TODO: deal with merge conflicts in an interactive way
                    ['fetch', '-p'],
                    ['checkout', '-B', 'develop', 'origin/develop'],
            ]:
                gcmd(git_command)

                changelog_path = os.path.join(
                    self._get_project_root(project), 'CHANGELOG.md',
                )
                helpers.update_changelog_file(changelog_path, release_notes, logger)

            for git_command in [
                    ['add', changelog_path],
                    # FIXME: make sure version number doesn't have `hotfix` in it, or does... just make it consistent
                    ['commit', '-m', f'Hotfix {version_name}'],
                    ['push', 'origin', 'develop'],
            ]:
                gcmd(git_command)

        rev_hash = self.get_rev_hash(project, 'develop')
        logger.info('%s complete adding release notes to develop %s (now %s)', project, version_name, rev_hash)
        return rev_hash

    def get_updated_repo_names(self, since_final: bool = True) -> List[str]:
        """
        Get a list of the full names of the repos that have commits to develop since either the last final or
        prerelease

        Note that when the develop branch is updated immediately after a release, it creates a merge commit, which is
        not counted for the purposes of this method.

        :param since_final: if True, look for updates since the latest final tag; otherwise, since latest prerelease
        """
        project_names = [project.name for project in self._get_updated_projects(self.projects, since_final)]
        logger.debug('Get updated projects: %s/%s since_final=%s', len(project_names), len(self.projects), since_final)
        return project_names

    def get_merge_logs(self, project: str) -> List[MergeLog]:
        """Get a list of namedtuples containing each merged PR, its Jira Key, and short SHA"""
        merges = []
        for log in self._get_merges_since(
                project,
                self.get_final_tag(project),
                '--pretty="%h %s"',
        ):
            sha, msg = log.split(' ', maxsplit=1)
            # like: 'Merge ATP-27_Test_seal to develop' and we want just 'ATP-27'
            try:
                key = msg.split(' ')[1].split('_', maxsplit=1)[0]
            except IndexError:  # just ignoring messages that don't fit the format
                logger.warning(
                    '%s: unexpected git log message format: "%s"; %s merge ignored from list',
                    project,
                    msg, sha,
                )
            merges.append(MergeLog(key, sha))
        logger.debug('%s get merge logs: %s merged PRs', project, len(merges))
        return merges

    def get_merge_count(self, project: str) -> int:
        """Get the number of merges to develop since the last final tag"""
        return self._get_merge_count_since(
            project,
            GitClient._get_latest_tag(
                project=GitClient.ProjectPath(self._get_project_root(project)),
                find_final=True,
            )
        )

    def get_prerelease_tag(self, project: str, min_version: Optional[TagData] = None) -> Optional['TagData']:
        """Get the latest prerelease tag name

        :param min_version: if included, will ignore all versions below this one
        :return: either the version string of the latest prerelease tag or `None` if one wasn't found
        """
        try:
            pre_tag = GitClient._get_latest_tag(
                project=GitClient.ProjectPath(self._get_project_root(project)),
                find_final=False,
            )
            if not min_version:
                logger.debug('%s get prerelease tag, min_version=None: %s', project, getattr(pre_tag, 'name', None))
                return pre_tag
        except AttributeError:
            logger.debug('%s get prerelease tag, min_version=%s: None', project, getattr(min_version, 'name', None))
            return None
        tag = pre_tag if GitClient.TagData.is_older_name(min_version.name, pre_tag.name) else None
        logger.debug('%s get prerelease tag, min_version=%s: %s', project, min_version, getattr(tag, 'name', None))
        return tag

    def get_compare_url(self, project: str) -> str:
        """Get the URL to compare the latest final with the latest prerelease on GitHub"""
        final = self.get_final_tag(project)
        return self._get_origin(project).compare(
            base=final.name,
            head=self.get_prerelease_tag(
                project,
                min_version=final,  # requires that a pre-release exists since last final
            ).name,
        ).html_url

    def get_merged_prs_url(self, project: str) -> str:
        """Get the URL to see merged PRs since the latest final on GitHub"""
        return GitClient._get_merged_prs_url(
            project,
            self.get_final_tag(project).date,
            datetime.now(timezone.utc).replace(microsecond=0).isoformat().split('+')[0],
        )

    # TODO: this is very Django specific, figure out less opinionated way for non-Django users
    def get_migration_count(self, project: str) -> int:
        """Get the number of new migration files since the latest final"""
        tag_name = self.get_final_tag(project).name
        return len(self._execute_project_git(
            project,
            [
                'diff', '--name-status', '--diff-filter=A',
                f'HEAD..{tag_name}', '--', 'src/**/migrations/',
            ]
        ).stdout.strip().splitlines())

    def get_final_tag(self, project: str) -> Optional['TagData']:
        """Get the latest final tag for a given project name"""
        return GitClient._get_latest_tag(
            project=GitClient.ProjectPath(self._get_project_root(project)),
            find_final=True,
        )

    def _get_origins(self, projects: List[str]) -> List[Repository]:
        """Get a list of the `origin` repos for each of the given project names

        http://developer.github.com/v3/repos/
        """
        return [self._get_origin(project) for project in projects]

    def _get_origin(self, project: str) -> Repository:
        """Get the remote repo that a project was cloned from

        Note: only a single remote, `origin`, is currently supported.
        """
        origin = self.github.get_repo(project)
        logger.debug('%s: get origin %s', project, origin.url)
        return origin

    def _is_updated_since(self, project: ProjectPath, since_final: bool = True) -> bool:
        """Check if the given origin has commits to develop since either the last final or prerelease"""
        return self._get_merge_count_since(
            project.name,
            GitClient._get_latest_tag(project, since_final)
        ) > 0

    def _get_updated_projects(self, projects: List[ProjectPath], since_final: bool = True) -> List['ProjectPath']:
        """Get a list of the `origin` repos that have commits to develop since either the last final or prerelease

        :param since_final: if True, look for updates since the latest final tag; otherwise, since latest prerelease
        """
        return [
            project for project in projects
            if self._is_updated_since(project, since_final)
        ]

    def _get_merge_count_since(self, project: str, tag: TagData) -> int:
        """Get the number of merges to develop since the given tag"""
        # The first result will be the merge commit from last release
        count = len(self._get_merges_since(project, tag,)) - 1
        logger.debug('%s merge count since %s: %s', project, tag.name, count)
        return count

    def _get_merges_since(self, project: str, tag: TagData, *flags: List[str]) -> List[str]:
        """Get the git log entries to develop since the given tag"""
        # FIXME: assumes master and developed have not diverged, which is not a safe assumption at all
        return self._execute_project_git(
            project,
            ['log', f'{tag.name}...origin/develop', '--merges', '--oneline', *flags]
        ).stdout.replace('"', '').splitlines()[:-1]  # remove last entry as it's the update from master

    def _fetch_tags(self, project: str) -> None:
        """Fetch the project's tags from origin to its local repo"""
        self._execute_project_git(
            project,
            ['fetch', '--tags'],
        )

    @staticmethod
    def _get_tag_lines(project: ProjectPath) -> List[str]:
        """Get stdout lines from `git tag`

        Each one should include commit SHA, tag name, and creator date.

        Some may be malformed as projects could have used different versioning schemes and tagging approaches in the
        past. This method does not filter them out, but at some point they should be filtered to be consistent with the
        versioning scheme used throughout this plugin.
        """
        tag_lines = _execute_path_git(
            project.path,
            ["tag", "--format=%(objectname:short) %(refname:short) %(creatordate:iso8601-strict)"]
        ).stdout.strip().splitlines()
        logger.debug(
            '%s: got %s tags from `git tag`',
            project.name,
            len(tag_lines),
        )
        # e.g. [['efdf7a56', 'v3.1.0', '2017-11-23T19:13:52+00:00'], ...]
        return [tag_line.split() for tag_line in tag_lines]

    @staticmethod
    def _get_tags(project: ProjectPath) -> List['TagTup']:
        """Get a project's tags from the local repo, not origin

        Returned list is sorted in descending order by version name using `packaging.version`.
        """
        tags = []
        tag_lines = GitClient._get_tag_lines(project)
        for unparsed_tag in tag_lines:
            # filters out "bad" tags and logs each one
            if GitClient._is_unparsed_tag_valid(project.name, unparsed_tag):
                tags.append(TagTup(*unparsed_tag))
                logger.debug('%s: successfully parsed tag %s', project.name, tags[-1].name)
        logger.debug('%s: %s/%s tags validated', project.name, len(tags), len(tag_lines))
        # sort using `packaging.version` with most recent first
        return sorted(tags, key=lambda tag: cached_parse(tag.name), reverse=True)

    def _backup_repo(self, project: str) -> str:
        """Create a backup of the entire local repo folder and return the destination

        :return: the dst path
        """
        # TODO: maybe it would be better to back up the whole repos root, instead of individual repos
        ref = self.get_ref(project)[:7]
        return helpers.copytree(
            self._get_project_root(project),
            self._get_backups_path(project),
            ref,
        )

    def _restore_repo(self, project: str, backup_path: str) -> str:
        """Restore a repo backup directory to its original location

        :param backup_path: absolute path to the backup's root as returned by `_backup_repo()`
        """
        # create a backup of the backup so it can be moved using the atomic `shutil.move`
        backup_swap = helpers.copytree(
            src=backup_path,
            dst_parent=self._get_backups_path(project),
            dst=self.get_ref(project)[:7] + '.swap',
        )
        # move the backup to the normal repo location
        project_root = self._get_project_root(project)
        shutil.rmtree(project_root)
        return shutil.move(src=backup_swap, dst=project_root)

    def _get_backups_path(self, project: str = None) -> str:
        """Get the backups dir path, either for all projects, or for the given project name"""
        return os.path.join(
            *[self.repos_root, '.backups']
            + ([project] if project else [])
        )

    def _execute_project_git(self, project: str, git_command: list) -> CompletedProcess:
        """Simple wrapper for executing git commands by project name"""
        return _execute_path_git(self._get_project_root(project), git_command)

    def _get_project_root(self, project: str) -> str:
        """Get the full path to the project root"""
        return GitClient._build_project_root(self.repos_root, project)

    @staticmethod
    def _build_project_root(repos_root: str, project: str) -> str:
        """Get the full path to the project's repo"""
        return os.path.join(repos_root, project)

    @classmethod
    def _get_latest_tag(cls, project: ProjectPath, find_final: bool = True) -> Optional['TagData']:
        """Get the latest final or prerelease tag

        Final tags do not contain a prerelease segment, but may contain a SemVer metadata segment.

        Tags are identified as prerelease tags if they contain a prerelease segment such as the following, where the
        hyphen-separated component (`rc.1`) makes up the prerelease segment:
        v1.0.0-rc.1
        v1.0.0-rc.1+sealed

        However, the presence of a SemVer metadata segment has no bearing on whether it's a prerelease tag or not.

        :param find_final: if True, look for the latest final tag; otherwise, look for latest prerelease
        """
        tag = cls._find_tag(
            project,
            cls.TagData.is_final_name if find_final
            else lambda tag: not cls.TagData.is_final_name(tag)
        )
        logger.debug('%s get latest tag (find_final=%s): %s', project.name, find_final, getattr(tag, 'name', None))
        return tag

    @classmethod
    def _find_tag(cls, project: ProjectPath, test: Callable[[str], bool]) -> Optional['TagData']:
        """Return the first tagtup that passes a given test or `None` if none found

        The order of the tags is important when using this method.
        """
        try:
            return cls.TagData(
                project,
                next((tagtup for tagtup in cls._get_tags(project) if test(tagtup.name)), None),
            )
        except ValueError:
            return None

    @staticmethod
    def _is_unparsed_tag_valid(project: str, unparsed_tag: List[str]) -> bool:
        def is_correct_field_length(tag_fields: List[str]) -> bool:
            if len(tag_fields) == len(TagTup._fields):
                return True
            logger.warning(
                '%s: The given tag_string (tagtup %s) contains %s fields, expected %s; excluding from list',
                project,
                tag_fields[1] if len(tag_fields) >= 2 else 'MISSINGTAG',
                len(tag_fields),
                len(TagTup._fields),
            )
            return False

        def is_correct_version_format(tag_fields: List[str]) -> bool:
            if tag_fields[1][:1] == 'v':
                return True
            logger.warning(
                (
                    '%s: The given tag_string (tagtup %s) contains a malformed '
                    'named, must start with "v"; excluding from list',
                ),
                project,
                tag_fields[1],
            )
            return False

        def is_parsable(tag_name: str):
            try:
                cached_parse(tag_name)
                return True
            except InvalidVersion:
                logger.warning(
                    (
                        '%s: the given tag_name (tagtup %s) could not be parsed '
                        'with `packaging.version.parse`; excluding from list'
                    ),
                    project,
                    tag_name,
                )
                return False

        return (
            is_correct_field_length(unparsed_tag)
            and is_correct_version_format(unparsed_tag)
            and is_parsable(unparsed_tag[1])
        )

    @staticmethod
    def _get_merged_prs_url(project: str, start_date: str, end_date: str) -> str:
        """Get the URL to see merged PRs in a date range on GitHub

        >>> GitClient._get_merged_prs_url('foo/bar-prj', '2018-01-01T22:02:39+00:00', '2018-01-02T22:02:39+00:00')[:46]
        'https://github.com/foo/bar-prj/pulls?utf8=✓&q='
        >>> GitClient._get_merged_prs_url('foo/bar-prj', '2018-01-01T22:02:39+00:00', '2018-01-02T22:02:39+00:00')[46:]
        'is:pr+is:closed+merged:2018-01-01T22:02:39+00:00..2018-01-02T22:02:39+00:00'
        """
        return f'{DOMAIN}/{project}/pulls?utf8=✓&q=is:pr+is:closed+merged:{start_date}..{end_date}'
