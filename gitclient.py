"""
This module provides high-level support for managing a local git repository and consolidating its changes with GitHub.
"""

import logging
import os
import sys
import subprocess

from typing import Callable, Union, List

from github import Github, PaginatedList
from github.Repository import Repository

import helpers

logger = logging.getLogger(__file__)


class GitCommandError(Exception):
    """A git command has failed"""


class GitClient:
    """Manage a local repo and its remote"""
    def __init__(self, config: dict):
        self.root = config['ROOT']
        self.new_version_name = config['NEW_VERSION_NAME']
        self.github_org = config['GITHUB_ORG']
        self.project_name = config['PROJECT_NAME']
        self.org = Github(config['GITHUB_TOKEN']).get_organization(
            self.github_org,
        )
        self.origin = self.org.get_repo(
            self.project_name,
        )

    def execute_command(self, git_command: list):
        """Execute a git command"""
        try:
            return helpers.run_subprocess(
                ['git'] + git_command,
                cwd=self.root,
            )
        except subprocess.CalledProcessError:
            logger.exception(
                'Failed git command=%s, output=%s',
                git_command,
                sys.exc_info()[1].stdout,
            )
            raise GitCommandError()

    def create_tag(self):
        """Create a git tag"""
        self.execute_command(
            ['tag', '-s', 'v' + self.new_version_name, '-m', 'v' + self.new_version_name, ]
        )

    def create_ref(self):
        """Create a ref and push it to origin

        https://developer.github.com/v3/git/refs/#create-a-reference
        """
        self.origin.create_git_ref(
            'refs/tags/{}'.format('v' + self.new_version_name),
            self.get_rev_hash('master'),  # TODO: this will have to be something else for hotfixes
        )

    def create_release(self, release_notes: str):
        """Create a GitHub release object and push it origin

        https://developer.github.com/v3/repos/releases/#create-a-release
        """
        self.origin.create_git_release(
            tag='v' + self.new_version_name,
            name='{} - Version {}'.format(self.project_name, self.new_version_name),
            message=release_notes,
            draft=False,
            prerelease=False,
        )

    def get_rev_hash(self, ref: str) -> str:
        """Get the SHA1 hash of a particular git ref"""
        return self.execute_command(
            ['rev-parse', ref]
        ).stdout.strip()  # Get rid of the newline character at the end

    def merge_and_create_release_commit(self, release_notes: str, changelog_path: str):
        """Create a release commit based on origin/develop and merge it to master"""
        for git_command in [
                # TODO: deal with merge conflicts in an interactive way
                ['fetch', '-p'],
                ['checkout', '-B', 'release-{}'.format(self.new_version_name), 'origin/develop'],
        ]:
            self.execute_command(git_command)

        helpers.update_changelog_file(
            changelog_path,
            release_notes,
            logger,
        )

        for git_command in [
                ['add', changelog_path],
                ['commit', '-m', 'Release {}'.format(self.new_version_name)],
                ['checkout', '-B', 'master', 'origin/master'],
                ['merge', '--no-ff', '--no-edit', 'release-{}'.format(self.new_version_name)],
                ['push', 'origin', 'master'],
        ]:
            self.execute_command(git_command)

    def update_develop(self):
        """Merge master branch to develop"""
        for git_command in [
                ['fetch', '-p'],
                ['checkout', '-B', 'develop', 'origin/develop'],
                ['merge', '--no-ff', '--no-edit', 'origin/master'],
                ['push', 'origin', 'develop'],
        ]:
            self.execute_command(git_command)

    @property
    def release_url(self) -> str:
        """Get the GitHub release URL"""
        return 'https://github.com/{github_org}/{project_name}/releases/tag/{new_version_name}'.format(
            github_org=self.github_org,
            project_name=self.project_name,
            new_version_name='v' + self.new_version_name,
        )

    def add_release_notes_to_develop(self, release_notes: str) -> str:
        """Wrap subprocess calls with some project-specific defaults.

        :return: Release commit hash.
        """
        for git_command in [
                # TODO: deal with merge conflicts in an interactive way
                ['fetch', '-p'],
                ['checkout', '-B', 'develop', 'origin/develop'],
        ]:
            self.execute_command(git_command)

            changelog_path = os.path.join(self.root, 'CHANGELOG.md')
            helpers.update_changelog_file(changelog_path, release_notes, logger)

        for git_command in [
                ['add', changelog_path],
                # FIXME: make sure version number doesn't have `hotfix` in it, or does... just make it consistent
                ['commit', '-m', 'Hotfix {}'.format(self.new_version_name)],
                ['push', 'origin', 'develop'],
        ]:
            self.execute_command(git_command)

        return helpers.run_subprocess(
            ['git', 'rev-parse', 'develop'],
            cwd=self.root,
        ).stdout.strip()  # Get rid of the newline character at the end

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
        return [self.org.get_repo(project_name) for project_name in project_names]

    def is_updated_since_last_final(self, repo: Repository) -> bool:
        """Check if the given repo has commits to develop since the last final release"""
        return self.count_merges_since(
            GitClient.get_latest_final_tag(repo).name
        ) > 0

    def get_updated_repos(self, project_names: List[str]) -> List[Repository]:
        """Get a list of repos that have commits to develop since the last final release"""
        return [
            repo for repo in self.get_all_repos(project_names)
            if self.is_updated_since_last_final(repo)
        ]

    def count_merges_since(self, tag_name: str) -> int:
        """Get the number of merges to develop since the given tag"""
        # FIXME: assumes master and developed have not diverged, which is not a safe assumption at all
        return len(self.execute_command(
            ['log', f'{tag_name}...develop', '--merges', '--oneline', ]
        ).stdout.splitlines()) - 1  # The first result will be the merge commit from last release
