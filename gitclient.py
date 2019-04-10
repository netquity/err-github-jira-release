"""
This module provides high-level support for managing local git repositories and consolidating changes with their
respective GitHub remotes.
"""

import logging
import os
import shutil
import sys
from collections import namedtuple
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache, total_ordering
from subprocess import CompletedProcess, CalledProcessError
from typing import Callable, List, Generator, Optional, NamedTuple

from github import Github
from packaging.version import parse, InvalidVersion
import semver

import helpers

logger = logging.getLogger(os.path.basename(__file__))
MergeLog = namedtuple('MergeLog', ['key', 'sha'])  # jira ticket key and commit sha
Tag = namedtuple('Tag', ['sha', 'name', 'date'])

DOMAIN = 'https://github.com'


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


def _execute_path_git(repo_root: str, git_command: list) -> CompletedProcess:
    """Execute a git command in a specific directory"""
    try:
        return helpers.run_subprocess(
            ['git'] + git_command,
            cwd=repo_root,
        )
    except CalledProcessError:
        logger.exception(
            '%s: Failed git command=%s, output=%s',
            repo_root,
            git_command,
            sys.exc_info()[1].stdout,
        )
        raise GitCommandError()


@total_ordering
class RepoTag:
    """A simple wrapper that combines a Repo with a Tag"""
    __slots__ = ['_repo', '_tag']

    def __init__(
            self,
            repo: 'Repo',
            tag: NamedTuple('Tag', [
                ('sha', str),
                ('name', str),
                ('date', str)
            ])
    ):
        if not isinstance(tag, Tag):
            raise TypeError(
                f'Inappropriate type: `tag` argument must be of type `Tag` but got `{type(tag)}`'
            )
        self._repo = repo
        self._tag = tag
        logger.debug('%s inited RepoTag for %s', repo.name, self.name)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.sha}, {self.name})'

    def __eq__(self, other) -> bool:
        RepoTag._validate_project_names(self.repo.name, other.repo.name)
        return self.name == other.name and self.date == other.date

    def __ne__(self, other) -> bool:
        RepoTag._validate_project_names(self.repo.name, other.repo.name)
        return self.name != other.name and self.date != other.date

    def __gt__(self, other) -> bool:
        RepoTag._validate_project_names(self.repo.name, other.repo.name)
        return semver.match(self.name[1:], f">{other.name[1:]}")  # removing leading `v`

    @staticmethod
    def _validate_project_names(name1, name2) -> None:
        if name1 != name2:
            raise ValueError('Cannot compare tags from different repos: %s vs. %s' % name1, name2)

    @property
    def repo(self):
        """Get the name of the repo this tag belongs to"""
        return self._repo

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
        return f'{DOMAIN}/{self.repo.name}/releases/tag/{self.name}'

    @property
    def date(self):
        """Get the creatordate of the tag"""
        return self._tag.date

    @staticmethod
    def is_final_name(tag_name: str) -> bool:
        """Determine whether the given tag string is a final tag string

        >>> RepoTag.is_final_name('v1.0.0')
        True
        >>> RepoTag.is_final_name('v1.0.0-rc.1')
        False
        >>> RepoTag.is_final_name('v1.0.0-rc.1+sealed')
        False
        >>> RepoTag.is_final_name('v1.0.0+20130313144700')
        True
        """
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
        >>> RepoTag.is_older_name('v1.0.0', 'v2.0.0')
        True
        >>> RepoTag.is_older_name('v1.0.0', 'v1.0.0')
        False
        >>> RepoTag.is_older_name('v1.0.0', 'v1.0.0-rc.1')
        False
        >>> RepoTag.is_older_name('v1.0.0', 'v1.0.1-rc.1+sealed')
        True
        >>> RepoTag.is_older_name('1.0.0', '2.0.0')  # need to include the leading `v`
        Traceback (most recent call last):
            ...
        ValueError: .0.0 is not valid SemVer string
        """
        return semver.match(old_tag_name[1:], f"<{new_tag_name[1:]}")

    @staticmethod
    def is_unparsed_tag_valid(repo: 'Repo', unparsed_tag: List[str]) -> bool:
        """Check whether the given unparsed tag is valid

        An unparsed tag will be a list of strings pulled from the `git log` output, like:
        ['efdf7a56', 'v3.1.0', '2017-11-23T19:13:52+00:00']
        """
        def is_correct_field_length(tag_fields: List[str]) -> bool:
            if len(tag_fields) == len(Tag._fields):
                return True
            logger.warning(
                '%s: The given tag_string (tag %s) contains %s fields, expected %s; excluding from list',
                repo.name,
                tag_fields[1] if len(tag_fields) >= 2 else 'MISSINGTAG',
                len(tag_fields),
                len(Tag._fields),
            )
            return False

        def is_correct_version_format(tag_fields: List[str]) -> bool:
            if tag_fields[1][:1] == 'v':
                return True
            logger.warning(
                (
                    '%s: The given tag_string (tag %s) contains a malformed '
                    'named, must start with "v"; excluding from list',
                ),
                repo.name,
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
                        '%s: the given tag_name (tag %s) could not be parsed '
                        'with `packaging.version.parse`; excluding from list'
                    ),
                    repo.name,
                    tag_name,
                )
                return False

        return (
            is_correct_field_length(unparsed_tag)
            and is_correct_version_format(unparsed_tag)
            and is_parsable(unparsed_tag[1])
        )


class RepoManager:
    """Manage local repos and their remotes

    Conventions:
        - `origin` is used to refer to the remote repo that a project was originally cloned from; there is only one per
          repo
        - `repo` is used to refer to the local repo that was cloned from origin
        - Public methods are always dealing with the latest changes. For example, `get_ref` will always get the most
          recent revision hash from the reflog; `get_merged_prs_url` will get show merged PRs since the latest final
          release.
    """
    __slots__ = ['repos_root', 'repos', 'github']

    def __init__(self, config: dict):
        self.github = Github(config['GITHUB_TOKEN'])
        self.repos_root = config['REPOS_ROOT']
        self.repos = [
            Repo(
                os.path.join(self.repos_root, project_name),
                self.github.get_repo(project_name),
                project_name,
            )
            for project_name in config['PROJECT_NAMES']
        ]

    def get_repos_in_stage(self, stage: helpers.Stages) -> Optional[List['Repo']]:
        """Get a list of repos for which the most recent tag is in the given stage

        For the first stage, we need all repos that have commits since the last final.

        If *not* at the beginning of a release cycle, we need all repos that were successfully transitioned into the
        most recent stage. i.e. if we're ready to sign/finalize a release, we do that with all the repos that were sent
        (which is a stage name) to UAT
        """
        def is_updated_since_final(repo):
            """At the seal stage, only get repos which have any updates at all since last final"""
            return repo.is_updated_since(since_final=True)

        def is_in_correct_stage(repo):
            """Other release stages require only repos in a particular stage"""
            return lowered_stage in Repo.get_latest_tag(repo, False).name.lower()

        if stage == helpers.Stages.SEALED:
            test = is_updated_since_final
        else:
            lowered_stage = helpers.Stages(stage.value - 1).verb.lower()
            test = is_in_correct_stage

        repos = [repo for repo in self.repos if test(repo)]
        logger.debug('Got repos in stage=%s: %s/%s updated', stage.name, len(repos), len(self.repos))
        return repos


class Repo(namedtuple('Repo', ['path', 'github', 'name'])):
    """Repos have a path and a limited set of attributes that can be derived from it"""
    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'

    @property
    def repos_root(self):
        """Get the path to the directory which stores all the repos

        Every repo has a `username/repo-name` directory structure, where `username` may be an organization like
        `netquity/err-github-jira-release'. This means that while the returned value does contain all the repos, the
        actual repos are always 2 levels in.
        """
        from os.path import dirname
        return dirname(dirname(self.path))

    @contextmanager
    def _gcmd(self) -> Generator[CompletedProcess, None, None]:
        """A context manager for interacting with local git repos in a safer way

        Records reflog position before doing anything and resets to this position if any of the git commands fail.
        """
        initial_ref = self.ref
        backup_path = self._backup_repo()
        try:
            yield lambda cmd: _execute_path_git(self.path, cmd)
        except GitCommandError as exc:
            self._restore_repo(backup_path)
            logger.error('%s: git commands failed; repo backup %s restored', self.name, initial_ref)
            raise exc

    @property
    def ref(self) -> str:
        """Get the latest rev hash from reflog"""
        ref = _execute_path_git(
            self.path,
            ['reflog', 'show', '--format=%H', '-1']
        ).stdout.splitlines()[0]
        logger.debug('%s get latest rev hash from reflog: %s', self.name, ref)
        return ref

    def create_tag(self, tag_name: str) -> None:
        """Create a git tag on whatever commit HEAD is pointing at"""
        tag_name = format_version(tag_name)
        _execute_path_git(
            self.path,
            ['tag', '-s', tag_name, '-m', tag_name, ]
        )
        logger.info('%s: created tag %s', self.name, tag_name)

    def get_rev_hash(self, ref: str) -> str:
        """Get the SHA1 hash of a particular git ref"""
        rev_hash = _execute_path_git(
            self.path,
            ['rev-parse', ref]
        ).stdout.strip()  # Get rid of the newline character at the end
        logger.debug('%s: get rev hash of ref %s: %s', self.name, ref, rev_hash)
        return rev_hash

    def get_merge_logs(self) -> List[MergeLog]:
        """Get a list of namedtuples containing each merged PR, its Jira Key, and short SHA"""
        merges = []
        for log in self._get_merges_since(
                self.get_final_tag(),
                '--pretty="%h %s"',
        ):
            sha, msg = log.split(' ', maxsplit=1)
            # like: 'Merge ATP-27_Test_seal to develop' and we want just 'ATP-27'
            try:
                key = msg.split(' ')[1].split('_', maxsplit=1)[0]
            except IndexError:  # just ignoring messages that don't fit the format
                logger.warning(
                    '%s: unexpected git log message format: "%s"; %s merge ignored from list',
                    self.name,
                    msg, sha,
                )
            merges.append(MergeLog(key, sha))
        logger.debug('%s get merge logs: %s merged PRs', self.name, len(merges))
        return merges

    def get_merge_count(self) -> int:
        """Get the number of merges to develop since the last final tag"""
        latest_tag = Repo.get_latest_tag(
            repo=self,
            find_final=True,
        )
        merge_count = self._get_merge_count_since(latest_tag)
        logger.debug('%s: %s merges to develop since %s', self.name, merge_count, latest_tag.name)
        return merge_count

    def get_prerelease_tag(self, min_version: Optional[RepoTag] = None) -> Optional[RepoTag]:
        """Get the latest prerelease tag name

        :param min_version: if included, will ignore all versions below this one
        :return: either the version string of the latest prerelease tag or `None` if one wasn't found
        """
        try:
            pre_tag = Repo.get_latest_tag(
                self,
                find_final=False,
            )
            if not min_version:
                logger.debug('%s get prerelease tag, min_version=None: %s', self.name, getattr(pre_tag, 'name', None))
                return pre_tag
        except AttributeError:
            logger.debug('%s get prerelease tag, min_version=%s: None', self.name, getattr(min_version, 'name', None))
            return None
        tag = pre_tag if RepoTag.is_older_name(min_version.name, pre_tag.name) else None
        logger.debug('%s get prerelease tag, min_version=%s: %s', self.name, min_version, getattr(tag, 'name', None))
        return tag

    def create_ref(self, version_name: str, ref: str = 'master') -> None:
        """Create a ref and push it to origin

        https://developer.github.com/v3/git/refs/#create-a-reference
        """
        version_name = format_version(version_name)
        self.github.create_git_ref(
            f'refs/tags/{version_name}',
            self.get_rev_hash(ref),  # TODO: this will have to be something else for hotfixes
        )
        logger.info('%s: created ref %s', self.name, version_name)

    def create_release(self, release_notes: str, version_name: str):
        """Create a GitHub release object and push it to origin

        https://developer.github.com/v3/repos/releases/#create-a-release
        """
        version_name = format_version(version_name)
        self.github.create_git_release(
            tag=version_name,
            name=f'{self.name} - Version {version_name}',
            message=release_notes,
            draft=False,
            prerelease=False,
        )
        logger.info('%s: created ref %s', self.name, version_name)

    def tag_develop(self, tag_name: str) -> None:
        """Compound check, tag, ref creation and pushing it all to origin

        :param tag_name: the new tag to apply to develop's HEAD
        """
        tag_name = format_version(tag_name)
        # FIXME: should wrap all commands with gcmd, rather than individually inside RepoManager code
        self.checkout_latest('develop')
        self.create_tag(tag_name)
        self.create_ref(tag_name, 'develop')

    def merge_and_create_release_commit(self, version_name: str, release_notes: str) -> None:
        """Create a release commit based on origin/develop and merge it to master"""
        logger.info('%s start merging and creating release commit for %s', self.name, version_name)
        with self._gcmd() as gcmd:
            for git_command in [
                    # TODO: deal with merge conflicts in an interactive way
                    ['fetch', '-p'],
                    ['checkout', '-B', f'release-{version_name}', 'origin/develop'],
            ]:
                gcmd(git_command)

            changelog_path = os.path.join(self.path, 'CHANGELOG.md',)
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
        logger.info('%s complete merging and creating release commit for %s', self.name, version_name)

    def update_develop(self) -> None:
        """Merge master branch to develop"""
        logger.debug('%s start update develop by merging master', self.name)
        with self._gcmd() as gcmd:
            for git_command in [
                    ['fetch', '-p'],
                    ['checkout', '-B', 'develop', 'origin/develop'],
                    ['merge', '--no-ff', '--no-edit', 'origin/master'],
                    ['push', 'origin', 'develop'],
            ]:
                gcmd(git_command)
        logger.debug('%s complete update develop by merging master', self.name)

    def checkout_latest(self, ref: str) -> None:
        """Check out the latest version of develop for the given repo"""
        with self._gcmd() as gcmd:
            for git_command in [
                    ['fetch', '--prune', '--force', '--tags'],
                    ['checkout', '-B', ref, f'origin/{ref}'],
            ]:
                gcmd(git_command)
        logger.info('%s: checked out latest origin/%s', self.name, ref)

    def add_release_notes_to_develop(self, version_name: str, release_notes: str) -> str:
        """Wrap subprocess calls with some repo-specific defaults.

        :return: Release commit hash.
        """
        logger.info(
            '%s start adding release notes to develop %s (now %s)',
            self.name,
            version_name,
            self.get_rev_hash('develop'),
        )
        with self._gcmd() as gcmd:
            for git_command in [
                    # TODO: deal with merge conflicts in an interactive way
                    ['fetch', '-p'],
                    ['checkout', '-B', 'develop', 'origin/develop'],
            ]:
                gcmd(git_command)

                changelog_path = os.path.join(self.path, 'CHANGELOG.md',)
                helpers.update_changelog_file(changelog_path, release_notes, logger)

            for git_command in [
                    ['add', changelog_path],
                    # FIXME: make sure version number doesn't have `hotfix` in it, or does... just make it consistent
                    ['commit', '-m', f'Hotfix {version_name}'],
                    ['push', 'origin', 'develop'],
            ]:
                gcmd(git_command)

        rev_hash = self.get_rev_hash('develop')
        logger.info('%s complete adding release notes to develop %s (now %s)', self.name, version_name, rev_hash)
        return rev_hash

    def get_compare_url(self) -> str:
        """Get the URL to compare versions on GitHub

        If the latest tag is a final, compare the last two finals; otherwise, compare the last prerelease with the
        latest final.
        """
        final = self.get_final_tag()
        latest = Repo.get_latest_tag(self)
        if final == latest:  # compare the last two final version
            final = next((tag for tag in Repo._get_final_tags(self) if tag.name != latest.name))
        logger.debug('%s: comparing base=%s and head=%s on GitHub', self.name, final.name, latest.name)
        return self.github.compare(base=final.name, head=latest.name).html_url

    def get_merged_prs_url(self) -> str:
        """Get the URL to see merged PRs since the latest final on GitHub"""
        return Repo._get_merged_prs_url(
            self,
            self.get_final_tag().date,
            datetime.now(timezone.utc).replace(microsecond=0).isoformat().split('+')[0],
        )

    # TODO: this is very Django specific, figure out less opinionated way for non-Django users
    def get_migration_count(self) -> int:
        """Get the number of new migration files added between last final and last """
        final_tag = self.get_final_tag()
        pre_tag = self.get_prerelease_tag(min_version=final_tag)
        if pre_tag is None or final_tag >= pre_tag:
            return 0
        return len(_execute_path_git(
            self.path,
            [
                'diff', '--name-status', '--diff-filter=A',
                f'{pre_tag.name}..{final_tag.name}', '--', 'src/**/migrations/',
            ]
        ).stdout.strip().splitlines())

    def get_final_tag(self) -> Optional['RepoTag']:
        """Get the latest final tag for a given repo name"""
        return Repo.get_latest_tag(
            repo=self,
            find_final=True,
        )

    def is_updated_since(self, since_final: bool = True) -> bool:
        """Check if the repo has commits to develop since either the last final or prerelease"""
        return self._get_merge_count_since(
            Repo.get_latest_tag(self, since_final)
        ) > 0

    def _get_merge_count_since(self, tag: RepoTag) -> int:
        """Get the number of merges to develop since the given tag"""
        # The first result will be the merge commit from last release
        count = len(self._get_merges_since(tag))
        logger.debug('%s merge count since %s: %s', self.name, tag.name, count)
        return count

    def _get_merges_since(self, tag: RepoTag, *flags: List[str]) -> List[str]:
        """Get the git log entries to develop since the given tag"""
        # FIXME: assumes master and developed have not diverged, which is not a safe assumption at all
        return _execute_path_git(
            self.path,
            ['log', f'{tag.name}...origin/develop', '--merges', '--oneline', *flags]
        ).stdout.replace('"', '').splitlines()[:-1]  # remove last entry as it's the update from master

    def _fetch_tags(self) -> None:
        """Fetch the repo's tags from origin to its local repo"""
        _execute_path_git(
            self.path,
            ['fetch', '--tags'],
        )

    def _backup_repo(self) -> str:
        """Create a backup of the entire local repo folder and return the destination

        :return: the dst path
        """
        # TODO: maybe it would be better to back up the whole repos root, instead of individual repos
        ref = self.ref[:7]
        return helpers.copytree(
            self.path,
            self._get_backups_path(self),
            ref,
        )

    def _restore_repo(self, backup_path: str) -> str:
        """Restore a repo backup directory to its original location

        :param backup_path: absolute path to the backup's root as returned by `_backup_repo()`
        """
        # create a backup of the backup so it can be moved using the atomic `shutil.move`
        backup_swap = helpers.copytree(
            src=backup_path,
            dst_parent=self._get_backups_path(self),
            dst=self.ref[:7] + '.swap',
        )
        # move the backup to the normal repo location
        shutil.rmtree(self.path)
        return shutil.move(src=backup_swap, dst=self.path)

    def _get_backups_path(self, repo: 'Repo' = None) -> str:
        """Get the backups dir path, either for all repos, or for the given repo name"""
        return os.path.join(
            *[self.repos_root, '.backups']
            + ([repo.name] if repo else [])
        )

    @staticmethod
    def _get_tag_lines(repo: 'Repo') -> List[str]:
        """Get stdout lines from `git tag`

        Each one should include commit SHA, tag name, and creator date.

        Some may be malformed as repos could have used different versioning schemes and tagging approaches in the
        past. This method does not filter them out, but at some point they should be filtered to be consistent with the
        versioning scheme used throughout this plugin.
        """
        tag_lines = _execute_path_git(
            repo.path,
            ["tag", "--format=%(objectname:short) %(refname:short) %(creatordate:iso8601-strict)"]
        ).stdout.strip().splitlines()
        logger.debug(
            '%s: got %s tags from `git tag`',
            repo.name,
            len(tag_lines),
        )
        # e.g. [['efdf7a56', 'v3.1.0', '2017-11-23T19:13:52+00:00'], ...]
        return [tag_line.split() for tag_line in tag_lines]

    @staticmethod
    def _get_tags(repo: 'Repo') -> List['Tag']:
        """Get a repo's tags from the local repo, not origin

        Returned list is sorted in descending order by version name using `packaging.version`.
        """
        tags = []
        tag_lines = Repo._get_tag_lines(repo)
        for unparsed_tag in tag_lines:
            # filters out "bad" tags and logs each one
            if RepoTag.is_unparsed_tag_valid(repo, unparsed_tag):
                tags.append(Tag(*unparsed_tag))
                logger.debug('%s: successfully parsed tag %s', repo.name, tags[-1].name)
        logger.debug('%s: %s/%s tags validated', repo.name, len(tags), len(tag_lines))
        # sort using `packaging.version` with most recent first
        return sorted(tags, key=lambda tag: cached_parse(tag.name), reverse=True)

    @staticmethod
    def _get_final_tags(repo: 'Repo') -> List['Tag']:
        """Get just the final tags i.e. filter out prerelease tags"""
        return [tag for tag in Repo._get_tags(repo) if RepoTag.is_final_name(tag.name)]

    @classmethod
    def get_latest_tag(cls, repo: 'Repo', find_final: bool = True) -> Optional[RepoTag]:
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
            repo,
            RepoTag.is_final_name if find_final
            else lambda tag: not RepoTag.is_final_name(tag)
        )
        logger.debug('%s get latest tag (find_final=%s): %s', repo.name, find_final, getattr(tag, 'name', None))
        return tag

    @classmethod
    def _find_tag(cls, repo: 'Repo', test: Callable[[str], bool]) -> Optional[RepoTag]:
        """Return the first tag that passes a given test or `None` if none found

        The order of the tags is important when using this method.
        """
        try:
            return RepoTag(
                repo,
                next((tag for tag in cls._get_tags(repo) if test(tag.name)), None),
            )
        except ValueError:
            return None

    @staticmethod
    def _get_merged_prs_url(repo: 'Repo', start_date: str, end_date: str) -> str:
        """Get the URL to see merged PRs in a date range on GitHub

        >>> RepoManager._get_merged_prs_url('foo/bar-prj', '2018-01-01T22:02:39+00:00', '2018-01-02T22:02:39+00:00')[:46]
        'https://github.com/foo/bar-prj/pulls?utf8=✓&q='
        >>> RepoManager._get_merged_prs_url('foo/bar-prj', '2018-01-01T22:02:39+00:00', '2018-01-02T22:02:39+00:00')[46:]
        'is:pr+is:closed+merged:2018-01-01T22:02:39+00:00..2018-01-02T22:02:39+00:00'
        """
        return f'{DOMAIN}/{repo.name}/pulls?utf8=✓&q=is:pr+is:closed+merged:{start_date}..{end_date}'
