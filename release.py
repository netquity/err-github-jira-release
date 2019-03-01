# coding: utf-8
"""An Errbot plugin for creating software releases by looking at completed work in Jira and GitHub."""
import errno
import logging
import os

from typing import List, Tuple, Mapping, Dict, Union

from errbot import BotPlugin, botcmd, arg_botcmd, ValidationException
from errbot.botplugin import recurse_check_structure
from errbot.backends.base import Message
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
        self.jira = JiraClient(self.get_jira_config())
        self.git = GitClient(self.get_git_config())

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
                    ['git', 'clone', f"git@github.com:{project_name}.git", project_name],
                    cwd=self.config['REPOS_ROOT'],
                )

    def get_configuration_template(self) -> str:
        return {
            'REPOS_ROOT': '/home/web/repos/',
            'JIRA_URL': None,
            'JIRA_USER': None,
            'JIRA_PASS': None,
            'GITHUB_TOKEN': None,
            'projects': {  # Map GitHub projects with their Jira keys
                # Full Name of the project in GitHub, like 'jakubroztocil/httpie', and the corresponding Jira key
                'project-full-name': 'PRJ',
            },
            'TEMPLATE_DIR': '/home/web/templates/',
            'changelog_path': '{}/CHANGELOG.md',
        }

    def check_configuration(self, configuration: Mapping) -> None:
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
        for _, v in projects_config.items():  # pylint: disable=invalid-name
            recurse_check_structure(projects_template['project-full-name'], v)

        configuration.update({'projects': projects_config})

    @arg_botcmd('--project-key', dest='project_key', type=str.upper, required=True)
    @arg_botcmd('--hotfix', action="store_true", dest='is_hotfix', default=False, required=False)
    def version(
            self,
            msg: Message,
            project_key: str,
            is_hotfix: bool,
    ) -> str:
        """Perform a version release to GitHub using issues from JIRA."""
        from gitclient import GitCommandError
        # Check out latest
        # TODO: check validity of given version number
        try:
            jira = JiraClient(self.get_jira_config())
            project_name = jira.get_project_name(project_key)

            release_type = jira.get_release_type(is_hotfix)
            new_jira_version = jira.create_version(project_key, release_type)
            jira.set_fix_version(
                project_key,
                new_jira_version.name,
                is_hotfix,
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
                project_key,
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
                        project_key,
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

    def _bump_project_tags(self, project_name: str, stage: str) -> str:
        """Tag the project's repo with the next version's tags and push to origin"""
        latest_final = self.git.get_latest_final_tag_name(project_name)
        project_key = self.get_project_key(project_name)
        new_version = self.jira.get_pending_version_name(
            project_key,
            stage,
            latest_final,
            self.git.get_latest_pre_release_tag_name(project_name, min_version=latest_final),
        )
        self.git.tag_develop(project_name, new_version)
        return new_version

    def _get_version_card(self, project_name: str) -> Dict:
        project_key = self.get_project_key(project_name)
        return {
            'Key': project_key,
            'Release Type': self.jira.get_release_type(project_key),

            'Previous Version': '<{url}|{tag}>'.format(
                url=self.git.get_latest_final_tag_url(project_name),
                tag=self.git.get_latest_final_tag_name(project_name),
            ),
            'Previous vCommit': self.git.get_latest_final_tag_sha(project_name)[:7],

            'Merge Count': self.git.get_merge_count(project_name),
            # TODO: it would be nice to be able to dynamically pass in functions for fields to show up on the card
            'New Migrations': self.git.get_migration_count(project_name),  # FIXME: too django-specific

            # To be removed for `fields`
            'GitHub Tag Comparison': self.git.get_latest_compare_url(project_name),
            # TODO: find a good public source for thumbnails; follow license
            'thumbnail': 'https://static.thenounproject.com/png/1662598-200.png',
        }

    @botcmd
    def seal(self, msg: Message, args) -> str:
        """Initiate the release sequence by tagging updated projects"""
        card_dict = {}
        for project_name in self.git.get_updated_repo_names(self.get_project_names()):
            # TODO: wrap in a try/except and roll back repos on any kind of failure
            new_version = self._bump_project_tags(project_name, helpers.Stages.SEALED)
            card_dict[project_name] = dict(
                self._get_version_card(project_name),
                **{'New Version Name': new_version}
            )

        yield f"{len(card_dict)} projects updated: \n\t• " + '\n\t• '.join(list(card_dict))

        for name, fields in card_dict.items():
            # CAUTION: Slack STRONGLY warns against sending more than 20 cards at a time:
            # https://api.slack.com/docs/message-attachments#attachment_limits
            self._send_version_card(
                msg,
                project_name=name,
                card_dict=fields,
            )

    def _get_project_root(self, project_name: str) -> str:
        """Get the root of the project's Git repo locally."""
        return self.config['REPOS_ROOT'] + project_name

    def get_project_names(self) -> List[str]:
        """Get the list of project names from the configuration"""
        return list(self.config['projects'])

    def get_project_key(self, project_name: str) -> str:
        """Get the Jira project key for the given project name"""
        # TODO: catch `KeyError`
        return self.config['projects'][project_name]

    def get_jira_config(self) -> dict:
        """Return data required for initializing JiraClient"""
        return {
            'URL': self.config['JIRA_URL'],
            'USER': self.config['JIRA_USER'],
            'PASS': self.config['JIRA_PASS'],
            'TEMPLATE_DIR': self.config['TEMPLATE_DIR'],
        }

    def get_git_config(self) -> dict:
        """Return data required for initializing GitClient"""
        return {
            'REPOS_ROOT': self.config['REPOS_ROOT'],
            'GITHUB_TOKEN': self.config['GITHUB_TOKEN'],
            'PROJECT_NAMES': self.get_project_names(),
        }

    def _send_version_card(self, message: Message, project_name: str, card_dict: Dict[str, Union[str, int]]) -> None:
        """Send the Slack card containing version set information

        :param message:
        :param project_name:
        :param card_dict: a dict of values to be displayed on the version card
        """
        return self.send_card(  # CAUTION: Slack STRONGLY warns against sending more than 20 cards at a time
            title=f'{project_name} - {card_dict.pop("New Version Name")}',
            link=card_dict.pop('GitHub Release URL'),
            in_reply_to=message,  # TODO: sometimes the message should be sent to a different channel
            thumbnail=card_dict.pop('thumbnail'),
            fields=tuple(card_dict.items()),
            color='green',
        )
