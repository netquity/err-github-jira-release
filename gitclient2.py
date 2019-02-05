"""
This module provides high-level support for managing local git repositories and consolidating changes with their
respective GitHub remotes.
"""

import logging
import os
import subprocess
import sys
from contextlib import contextmanager

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
            'Failed git command=%s, output=%s',
            git_command,
            sys.exc_info()[1].stdout,
        )
        raise GitCommandError()


class GitClient2:
    """Manage local repos and their remotes"""
    def __init__(self, config: dict):
        self.repos_root = config['REPOS_ROOT']
        self.project_names = config['PROJECT_NAMES']
        self.github = Github(config['GITHUB_TOKEN'])

    @contextmanager
    def _gcmd(self, project_name: str):  # TODO: add Generator hint
        """A context manager for interacting with local git repos in a safer way

        Records reflog position before doing anything and resets to this position if any of the git commands fail.
        """
        initial_ref = self.get_latest_ref(project_name)
        try:
            yield lambda cmd: self._execute_project_git(project_name, cmd)
        except GitCommandError as exc:
            self.reset_hard(project_name, initial_ref)
            logger.warning('')  # TODO
        finally:
            self.clean(project_name)

    def get_latest_ref(self, project_name: str) -> str:
        """Get the latest rev hash from reflog"""
        return self._execute_project_git(
            project_name,
            ['reflog', 'show', '--format=%H', '-1']
        ).stdout.splitlines()[0]

    def clean(self, project_name: str) -> subprocess.CompletedProcess:
        """Recursively remove files that aren't under source control"""
        return self._execute_project_git(self.get_project_root(project_name), ['clean', '-f'])

    def reset_hard(self, project_name: str, ref: str) -> subprocess.CompletedProcess:
        return self._execute_project_git(self.get_project_root(project_name), ['reset', '--hard', ref])

    def _execute_project_git(self, project_name: str, git_command: list) -> subprocess.CompletedProcess:
        """Simple wrapper for executing git commands by project name"""
        return _execute_path_git(self.get_project_root(project_name), git_command)

    def get_project_root(self, project_name: str) -> str:
        """Get the full path to the project root"""
        return os.path.join(self.repos_root, project_name)

    def create_tag(self, project_name: str, tag_name: str):
        """Create a git tag"""
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

    def _get_remote_repo(self, project_name: str):
        return self.github.get_repo(project_name)

    def get_rev_hash(self, project_name: str, ref: str) -> str:
        """Get the SHA1 hash of a particular git ref"""
        return self._execute_project_git(
            project_name,
            ['rev-parse', ref]
        ).stdout.strip()  # Get rid of the newline character at the end

    def merge_and_create_release_commit(
            self, project_name: str, new_version_name: str,
            release_notes: str, changelog_path: str
    ):
        """Create a release commit based on origin/develop and merge it to master"""
        for git_command in [
                # TODO: deal with merge conflicts in an interactive way
                ['fetch', '-p'],
                ['checkout', '-B', f'release-{new_version_name}', 'origin/develop'],
        ]:
            self._execute_project_git(project_name, git_command)

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
            self._execute_project_git(project_name, git_command)

    def update_develop(self, project_name: str) -> None:
        """Merge master branch to develop"""
        for git_command in [
                ['fetch', '-p'],
                ['checkout', '-B', 'develop', 'origin/develop'],
                ['merge', '--no-ff', '--no-edit', 'origin/master'],
                ['push', 'origin', 'develop'],
        ]:
            self._execute_project_git(project_name, git_command)

    @property
    def release_url(self, project_name: str, new_version_name: str) -> str:
        """Get the GitHub release URL"""
        return f'https://github.com/{project_name}/releases/tag/{new_version_name}'

    def add_release_notes_to_develop(self, project_name: str, new_version_name: str, release_notes: str) -> str:
        """Wrap subprocess calls with some project-specific defaults.

        :return: Release commit hash.
        """
        for git_command in [
                # TODO: deal with merge conflicts in an interactive way
                ['fetch', '-p'],
                ['checkout', '-B', 'develop', 'origin/develop'],
        ]:
            self._execute_project_git(project_name, git_command)

            changelog_path = os.path.join(
                self.get_project_root(project_name), 'CHANGELOG.md',
            )
            helpers.update_changelog_file(changelog_path, release_notes, logger)

        for git_command in [
                ['add', changelog_path],
                # FIXME: make sure version number doesn't have `hotfix` in it, or does... just make it consistent
                ['commit', '-m', f'Hotfix {new_version_name}'],
                ['push', 'origin', 'develop'],
        ]:
            self._execute_project_git(project_name, git_command)

        return self.get_rev_hash(project_name, 'develop')

    @classmethod
    def get_latest_pre_release_tag(cls, origin: Repository) -> Union['github.Tag.tag', None]:
        """Get the latest pre-release tag

        Tags are identified as pre-release tags if they contain a pre-release segment such as the following, where the
        hyphen-separated component (`rc.1`) makes up the pre-release segment:
        v1.0.0-rc.1
        v1.0.0-rc.1+sealed

        However, the presence of a SemVer metadata segment has no bearing on whether it's a pre-release tag or not.
        """
        return cls.find_tag(origin, cls.is_prerelease_tag_name)

    @classmethod
    def get_latest_final_tag(cls, origin: Repository) -> Union['github.Tag.tag', None]:
        """Get the latest final tag

        Final tags do not contain a pre-release segment, but may contain a SemVer metadata segment.
        """
        return cls.find_tag(origin, lambda tag: not cls.is_prerelease_tag_name(tag))

    @classmethod
    def find_tag(cls, origin: Repository, test: Callable[[str], bool]) -> Union['github.Tag.tag', None]:
        """Return the first tag that passes a given test or `None` if none found"""
        return next((tag for tag in cls.get_tags(origin) if test(tag.name)), None)

    @classmethod
    def is_prerelease_tag_name(cls, tag_name: str) -> bool:
        """Determine whether the given tag string is a pre-release tag string

        >>> GitClient.is_prerelease_tag_name('v1.0.0')
        false
        >>> GitClient.is_prerelease_tag_name('v1.0.0-rc.1')
        true
        >>> GitClient.is_prerelease_tag_name('v1.0.0-rc.1+sealed')
        true
        >>> GitClient.is_prerelease_tag_name('v1.0.0+20130313144700')
        false
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

    @classmethod
    def get_tags(cls, origin: Repository) -> PaginatedList.PaginatedList:
        """Get all the tags for the repo"""
        return origin.get_tags()  # TODO: consider searching local repo instead of GitHub

    def get_all_repos(self, project_names: List[str]) -> List[Repository]:
        """Get a list of all the repositories under management

        http://developer.github.com/v3/repos/
        """
        return [self._get_remote_repo(project_name) for project_name in project_names]

    def is_updated_since_last_final(self, repo: Repository) -> bool:
        """Check if the given repo has commits to develop since the last final release"""
        return self.count_merges_since(
            repo.full_name,
            GitClient2.get_latest_final_tag(repo).name
        ) > 0

    def get_updated_repos(self, project_names: List[str]) -> List[Repository]:
        """Get a list of repos that have commits to develop since the last final release"""
        return [
            repo for repo in self.get_all_repos(project_names)
            if self.is_updated_since_last_final(repo)
        ]

    def count_merges_since(self, project_name: str, tag_name: str) -> int:
        """Get the number of merges to develop since the given tag"""
        # FIXME: assumes master and developed have not diverged, which is not a safe assumption at all
        return len(self._execute_project_git(
            project_name,
            ['log', f'{tag_name}...develop', '--merges', '--oneline', ]
        ).stdout.splitlines()) - 1  # The first result will be the merge commit from last release
