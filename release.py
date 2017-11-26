# coding: utf-8
import errno
import logging
import os

from errbot import BotPlugin, arg_botcmd, ValidationException
from errbot.botplugin import recurse_check_structure
from gitclient import GitClient
from jiraclient import JiraClient
import utils

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

    Note: Currently does not support hotfix releases; only standard releases from origin/develop to origin/master are
    supported.
    """

    def activate(self):
        if not self.config:
            # Don't allow activation until we are configured
            message = 'Release is not configured, please do so.'
            self.log.info(message)
            self.warn_admins(message)
            return

        self.setup_repos()
        self.github_client = self.get_github_client()  # pylint:disable=attribute-defined-outside-init
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
                utils.run_subprocess(
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
        for k, v in projects_config.items():
            recurse_check_structure(projects_template['some-project'], v)

        configuration.update({'projects': projects_config})

    @arg_botcmd('--project-key', dest='project_key', type=str.upper, required=True)
    def version(
            self,
            msg: 'errbot.backends.base.Message',
            project_key: str,
    ) -> str:
        """Perform a version release to GitHub using issues from JIRA."""
        from gitclient import GitCommandError
        # Check out latest
        # TODO: check validity of given version number
        try:
            jira = JiraClient(self.jira_config)
            project_name = jira.get_project_name()
            git = GitClient(self.get_project_root(project_name), self.log)

            release_type = jira.get_release_type()
            new_jira_version = jira.create_version(release_type)
            jira.set_fix_version(
                new_jira_version.name,
            )
            release_notes = jira.get_release_notes(new_jira_version)
        except JIRAError:
            exc_message = jira.delete_version(
                project_key,
                new_jira_version,
            )
            self.log.exception(
                exc_message,
            )
            return exc_message

        try:
            git.merge_and_create_release_commit(
                new_jira_version.name,
                release_notes,
                self.config['changelog_path'].format(git.root)
            )
            commit_hash = git.get_rev_hash('master')
        except GitCommandError as exc:  # TODO: should the exception be changed to GitCommandError?
            self.log.exception(
                'Unable to merge release branch to master and create release commit.'
            )
            exc_message = jira.delete_version(
                project_key,
                new_jira_version,
                'git',
            )
            return exc_message

        repo = self.github_client.get_organization(
            self.config['projects'][project_name]['github_org'],
        ).get_repo(
            project_name,
        )

        git.create_tag(
            new_jira_version.name,
        )
        repo.create_git_ref(
            'refs/tags/{}'.format('v' + new_jira_version.name),
            commit_hash,
        )
        repo.create_git_release(
            tag='v' + new_jira_version.name,
            name='{} - Version {}'.format(project_name, new_jira_version.name),
            message=release_notes,
            draft=False,
            prerelease=False,
        )

        git.merge_master_to_develop()
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
                    Release.get_github_release_url(
                        self.config['projects'][project_name]['github_org'],
                        project_name,
                        'v' + new_jira_version.name,
                    ),
                ),
            ),
            color='green',
        )

    @staticmethod
    def get_github_release_url(github_org: str, project_name: str, new_version_name: str) -> str:
        return 'https://github.com/{github_org}/{project_name}/releases/tag/{new_version_name}'.format(
            github_org=github_org,
            project_name=project_name,
            new_version_name=new_version_name,
        )

    def get_project_root(self, project_name: str) -> str:
        """Get the root of the project's Git repo locally."""
        return self.config['REPOS_ROOT'] + project_name

    @property
    def jira_config(self):
        return {
            'URL': self.config['JIRA_URL'],
            'USER': self.config['JIRA_USER'],
            'PASS': self.config['JIRA_PASS'],
            'PROJECT_KEY': self.config['PROJECT_KEY'],
            'TEMPLATE_DIR': self.config['TEMPLATE_DIR'],
        }

    def get_github_client(self) -> Github:
        """Get an instance of the PyGitHub client using the plugins configuration for authentication."""
        return Github(self.config['GITHUB_TOKEN'])
