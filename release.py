# coding: utf-8
import errno
import logging
import os

from errbot import BotPlugin, arg_botcmd, ValidationException
from errbot.botplugin import recurse_check_structure
from gitclient import GitClient
from jiraclient import JiraClient

import helpers

logger = logging.getLogger(__file__)

try:
    from jira import JIRA, JIRAError
    from github import Github
except ImportError:
    logger.error("Please install 'jira' and 'pygithub' Python packages")


class Release(BotPlugin):  # pylint:disable=too-many-ancestors
    """Perform version releases between JIRA and GitHub.

    For a given JIRA project key, produce a tagged release on GitHub. Produce the version by taking all of JIRA tickets
    under the given project key where status=closed and there is no `Fix Version`. Inspect all of the tickets in that
    set to determine the highest `Release Type` value, where choices are [Major,Minor,Patch], in accordance with
    semver.org.  Use the highest found `Release Type` value to decide the version bump from the last released version
    for that project.

    Example:
        Project Key: FOO
        Last Release: 1.2.3
        Tickets closed since last release:
            FOO-100: Minor
            FOO-101: Patch
            FOO-102: Major  <-- the highest `Release Type` since the last release

        FOO-102 determines that the release is a major version bump; the result is FOO v2.0.0. FOO-{100,101,102} all
        get updated with `Fix Version = 2.0.0`. The release notes are added to the CHANGELOG.md file and to the tag
        created at GitHub.

    Hotfixes are supported only under the following workflow:
    When a hotfix is required, a new branch must be created based on the master branch. All hotfix related work is to go
    into that branch. Upon completion and testing, the hotfix branch is deployed to the production server. A commit with
    the hotfix changelog is pushed to develop for posterity, but no other changes are incorporated.  Only one hotfix is
    allowed per release at the moment. The hotfix branch must:
        - be titled `hotfix`
        - NEVER get merged into any branch (neither master nor develop)
        - be discarded prior to the next standard release
        - not contain any database migrations
    """

    def activate(self):
        if not self.config:
            # Don't allow activation until we are configured
            message = 'Release is not configured, please do so.'
            self.log.info(message)
            self.warn_admins(message)
            return

        self.setup_repos()
        super().activate()

    def setup_repos(self):
        """Clone the projects in the configuration into the `REPOS_ROOT` if they do not exist already."""
        try:
            os.makedirs(self.config['REPOS_ROOT'])
        except OSError as exc:
            # If the error is that the directory already exists, we don't care about it
            if exc.errno != errno.EEXIST:
                raise exc

        for project_name in self.config['projects']:
            if not os.path.exists(os.path.join(self.config['REPOS_ROOT'], project_name)):
                # Possible race condition if folder somehow gets created between check and creation
                helpers.run_subprocess(
                    ['git', 'clone', self.config['projects'][project_name]['repo_url']],
                    cwd=self.config['REPOS_ROOT'],
                )

    def get_configuration_template(self) -> str:
        return {
            'REPOS_ROOT': '/home/web/repos/',
            'JIRA_URL': None,
            'JIRA_USER': None,
            'JIRA_PASS': None,
            'GITHUB_TOKEN': None,
            'projects': {
                'some-project': {  # Name of the project in GitHub
                    'jira_key': 'PRJ',
                    'repo_url': 'git@github.com:netquity/some-project.git',
                    'github_org': 'netquity',
                },
            },
            'TEMPLATE_DIR': '/home/web/templates/',
            'changelog_path': '{}/CHANGELOG.md',
        }

    def check_configuration(self, configuration: 'typing.Mapping') -> None:
        """Allow for the `projects` key to have a variable number of definitions."""
        # Remove the `projects` key from both the template and the configuration and then test them separately
        try:
            config_template = self.get_configuration_template().copy()
            projects_template = config_template.pop('projects')
            projects_config = configuration.pop('projects')  # Might fail
        except KeyError:
            raise ValidationException(
                'Your configuration must include a projects key with at least one project configured.'
            )

        recurse_check_structure(config_template, configuration)

        # Check that each project configuration matches the template
        for _, v in projects_config.items():
            recurse_check_structure(projects_template['some-project'], v)

        configuration.update({'projects': projects_config})

    @arg_botcmd('--project-key', dest='project_key', type=str.upper, required=True)
    @arg_botcmd('--hotfix', action="store_true", dest='is_hotfix', default=False, required=False)
    def version(
            self,
            msg: 'errbot.backends.base.Message',
            project_key: str,
            is_hotfix: bool,
    ) -> str:
        """Perform a version release to GitHub using issues from JIRA."""
        from gitclient import GitCommandError
        # Check out latest
        # TODO: check validity of given version number
        try:
            jira = JiraClient(self.get_jira_config(project_key))
            project_name = jira.get_project_name()

            release_type = jira.get_release_type(is_hotfix)
            new_jira_version = jira.create_version(release_type)
            jira.set_fix_version(
                new_jira_version.name,
                is_hotfix,
            )
            release_notes = jira.get_release_notes(new_jira_version)
        except JIRAError:
            exc_message = jira.delete_version(
                new_jira_version,
            )
            self.log.exception(
                exc_message,
            )
            return exc_message

        try:
            git = GitClient(self.get_git_config(project_name, new_jira_version.name))
            if is_hotfix:
                git.add_release_notes_to_develop(release_notes)  # TODO: need better exception handling
            else:
                git.merge_and_create_release_commit(
                    release_notes,
                    self.config['changelog_path'].format(git.root)
                )
        except GitCommandError as exc:  # TODO: should the exception be changed to GitCommandError?
            self.log.exception(
                'Unable to merge release branch to master and create release commit.'
            )
            exc_message = jira.delete_version(
                new_jira_version,
                'git',
            )
            return exc_message

        git.create_tag()
        git.create_ref()
        git.create_release(release_notes)
        if not is_hotfix:
            git.update_develop()

        return self.send_card(
            in_reply_to=msg,
            summary='I was able to complete the %s release for you.' % project_name,
            fields=(
                ('Project Key', project_key),
                ('New Version', 'v' + new_jira_version.name),
                ('Release Type', release_type),
                (
                    'JIRA Release',
                    jira.get_release_url(
                        new_jira_version.id,
                    ),
                ),
                (
                    'GitHub Release',
                    git.release_url,
                ),
            ),
            color='green',
        )

    def get_project_root(self, project_name: str) -> str:
        """Get the root of the project's Git repo locally."""
        return self.config['REPOS_ROOT'] + project_name

    def get_jira_config(self, project_key: str) -> dict:
        """Return data required for initializing JiraClient"""
        return {
            'URL': self.config['JIRA_URL'],
            'USER': self.config['JIRA_USER'],
            'PASS': self.config['JIRA_PASS'],
            'PROJECT_KEY': project_key,
            'TEMPLATE_DIR': self.config['TEMPLATE_DIR'],
        }

    def get_git_config(self, project_name: str, new_version_name: str) -> dict:
        """Return data required for initializing GitClient"""
        return {
            'ROOT': self.get_project_root(project_name),
            'PROJECT_NAME': project_name,
            'GITHUB_ORG': self.config['projects'][project_name]['github_org'],
            'GITHUB_TOKEN': self.config['GITHUB_TOKEN'],
            'NEW_VERSION_NAME': new_version_name,
        }
