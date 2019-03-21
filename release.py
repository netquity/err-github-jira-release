# coding: utf-8
"""An Errbot plugin for creating software releases by looking at completed work in Jira and GitHub."""
import errno
import logging
import os

from typing import List, Mapping, Dict, Union, Optional

from errbot import BotPlugin, botcmd, arg_botcmd, ValidationException
from errbot.botplugin import recurse_check_structure
from errbot.backends.base import Message
from gitclient import GitClient
from jiraclient import JiraClient, NoJIRAIssuesFoundError

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

        self._setup_repos()
        super().activate()
        self.jira = JiraClient(self._get_jira_config())  # pylint:disable=attribute-defined-outside-init
        self.git = GitClient(self._get_git_config())  # pylint:disable=attribute-defined-outside-init

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
            'UAT_CHANNEL_IDENTIFIER': '#uat',  # certain messages will be sent here instead of as a reply
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

    @botcmd
    def seal(self, msg: Message, args) -> Optional[str]:  # pylint:disable=unused-argument
        """Initiate the release sequence by tagging updated projects"""
        card_dict = {}
        for project_name in self.git.get_updated_repo_names(self._get_project_names(), since_final=False):
            # TODO: wrap in a try/except and roll back repos on any kind of failure
            # TODO: these bumps can all be done asynchronously, they don't depend on each other
            try:
                new_version = self._bump_repo_tags(project_name, helpers.Stages.SEALED)
                card_dict[project_name] = dict(
                    self._get_version_card(project_name),
                    **{'New Version Name': new_version}
                )
            except NoJIRAIssuesFoundError as exc:
                key = self._get_project_key(project_name)
                failure_message = (  # TODO: consider putting this information in card fields instead
                    'Since `{final}`, {project_name} '
                    'had {merge_summary} '
                    'but <{jira_issues}|Jira> {exc_msg}.'
                ).format(
                    final=self.git.get_final_tag(project_name).name,
                    project_name=project_name,
                    merge_summary=self._get_merge_summary(project_name),
                    jira_issues=self.jira.get_latest_issues_url(key),
                    exc_msg=str(exc)[0].lower() + str(exc)[1:],
                )

                self.log.exception(
                    failure_message,
                )
                self.send_card(
                    in_reply_to=msg,
                    body=failure_message,
                    color='red',
                )
        if not card_dict:
            return self.send_card(
                in_reply_to=msg,
                body='No projects updated.',
                color='red',
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

        return f'{len(card_dict)} / {len(self._get_project_names())} projects updated since last release.'

    @botcmd
    def send(self, msg: Message, args) -> str:  # pylint:disable=unused-argument
        """Update SemVer metadata and tell the configured UAT channel that testing can begin

        This command is called only after the `seal` command is called and the sealed version set is tested and
        approved.
        """
        fields = ()
        updated_projects = self.git.get_updated_repo_names(self._get_project_names())
        for project_name in updated_projects:
            try:  # TODO: need similar try/except wherever calling bump_project_tags, but keep it DRY
                failure_message = ''
                new_version = self._bump_repo_tags(project_name, helpers.Stages.SENT)
                # form a field for each project formatted like:
                # ('net-net - v10.0.0 → v11.0.0-rc.2', '<https://best-url.com|12 PRs (major)>')
                fields += (
                    '{repo_name} - {final} → {pre}'.format(  # field title
                        repo_name=project_name.split("/")[1],  # get rid of org name for brevity
                        final=self.git.get_final_tag(project_name).name,
                        pre=new_version,
                    ),
                    self._get_merge_summary(project_name)
                    + f' ({self.jira.get_release_type(self._get_project_key(project_name))})',
                ),
            except helpers.InvalidStageTransitionError:
                failure_message = f'Invalid stage transition attempted when bumping {project_name}'
            except helpers.InvalidVersionNameError:
                failure_message = f'Invalid pre_version given when bumping {project_name}'
            finally:
                if failure_message:
                    self.log.exception(
                        failure_message,
                    )
                    return self.send_card(  # pylint:disable=lost-exception
                        in_reply_to=msg,
                        body=failure_message,
                        color='red',
                    )
        self.send_card(  # CAUTION: Slack STRONGLY warns against sending more than 20 cards at a time
            title=str(len(updated_projects)) + ' release(s)',
            to=self.build_identifier(self.config['UAT_CHANNEL_IDENTIFIER']),
            fields=fields,
            color='green',
        )
        return "I have sent your sealed version set to the UAT channel. Awaiting their approval."

    @botcmd
    def sign(self, msg: Message, args):
        from gitclient import GitCommandError
        fields = ()
        updated_projects = self.git.get_updated_repo_names(self._get_project_names())
        for project_name in updated_projects:
            with self.git.project_git(project_name) as git:
                key = self._get_project_key(project_name)
                final = git.get_final_tag()
                # new_version_name = self._bump_repo_tags(project_name, helpers.Stages.SIGNED)  # NOTE: comes with no `v`
                new_version_name = self.jira.get_pending_version_name(
                    key,
                    helpers.Stages.SIGNED,
                    git.get_rev_hash(ref="origin/develop")[:7],
                    final.name,
                    git.get_prerelease_tag(min_version=final).name,
                )
                jira_version = self.jira.create_version(
                    key,
                    new_version_name,
                    released=True,
                )
                self.jira.set_fix_version(
                    key,
                    jira_version.name,
                    is_hotfix=False,  # FIXME: need to actually do something for hotfixes
                )
                release_notes = self.jira.get_release_notes(jira_version, git.get_merge_logs())
                is_hotfix = False  # TODO: TEMPORARY SHIM; REMOVE!!!
                try:
                    if is_hotfix:
                        # TODO: need better exception handling
                        git.add_release_notes_to_develop(new_version_name, release_notes)
                    else:
                        # FIXME: need to get THIS sha, not the one of most recent commit earlier
                        git.merge_and_create_release_commit(
                            new_version_name=new_version_name,
                            release_notes=release_notes,
                        )
                except GitCommandError as exc:  # TODO: should the exception be changed to GitCommandError?
                    self.log.exception(
                        'Unable to merge release branch to master and create release commit.'
                    )
                    exc_message = JiraClient.delete_version(
                        key,
                        jira_version,
                        'git',
                    )
                    return exc_message

                # Need the merge commit sha as part of the version metadata
                new_version_name = helpers.change_sha(new_version_name, git.get_ref()[:7])
                self.jira.change_version_name(jira_version, new_version_name)
                git.create_tag(tag_name=new_version_name)
                git.create_ref(new_version_name=new_version_name)  # FIXME: reference already exists?
                git.create_release(release_notes=release_notes, new_version_name=new_version_name)
                if not is_hotfix:
                    git.update_develop()

    def _get_merge_summary(self, project_name: str) -> str:
        """Return a link to GitHub's issue search showing the merged PRs """
        return '<{url}|{pr_count} merged PR(s)>'.format(
            url=self.git.get_merged_prs_url(project_name),
            pr_count=self.git.get_merge_count(project_name),
        )

    def _get_project_root(self, project_name: str) -> str:
        """Get the root of the project's Git repo locally."""
        return self.config['REPOS_ROOT'] + project_name

    def _get_project_names(self) -> List[str]:
        """Get the list of project names from the configuration"""
        return list(self.config['projects'])

    def _get_project_key(self, project_name: str) -> str:
        """Get the Jira project key for the given project name"""
        # TODO: catch `KeyError`
        return self.config['projects'][project_name]

    def _get_jira_config(self) -> dict:
        """Return data required for initializing JiraClient"""
        return {
            'URL': self.config['JIRA_URL'],
            'USER': self.config['JIRA_USER'],
            'PASS': self.config['JIRA_PASS'],
            'TEMPLATE_DIR': self.config['TEMPLATE_DIR'],
        }

    def _get_git_config(self) -> dict:
        """Return data required for initializing GitClient"""
        return {
            'REPOS_ROOT': self.config['REPOS_ROOT'],
            'GITHUB_TOKEN': self.config['GITHUB_TOKEN'],
            'PROJECT_NAMES': self._get_project_names(),
        }

    def _get_version_card(self, project_name: str) -> Dict:
        with self.git.project_git(project_name) as git:
            final = git.get_final_tag()
            project_key = self._get_project_key(project_name)
            return {
                'Key': project_key,
                'Release Type': self.jira.get_release_type(project_key),

                'Previous Version': f'<{final.url}|{final.name}>',
                'Previous vCommit': final.sha,

                'Merge Count': git.get_merge_count(),
                # TODO: it would be nice to be able to dynamically pass in functions for fields to show up on the card
                'New Migrations': git.get_migration_count(),  # FIXME: too django-specific

                # To be removed for `fields`
                'GitHub Tag Comparison': git.get_compare_url(),
                # TODO: find a good public source for thumbnails; follow license
                'thumbnail': 'https://static.thenounproject.com/png/1662598-200.png',
            }

    def _send_version_card(self, message: Message, project_name: str, card_dict: Dict[str, Union[str, int]]) -> None:
        """Send the Slack card containing version set information

        :param message:
        :param project_name:
        :param card_dict: a dict of values to be displayed on the version card
        """
        return self.send_card(  # CAUTION: Slack STRONGLY warns against sending more than 20 cards at a time
            title=f'{project_name} - {card_dict.pop("New Version Name")}',
            link=card_dict.pop('GitHub Tag Comparison'),
            in_reply_to=message,  # TODO: sometimes the message should be sent to a different channel
            thumbnail=card_dict.pop('thumbnail'),
            fields=tuple(card_dict.items()),
            color='green',
        )

    def _bump_repo_tags(self, project_name: str, stage: str) -> str:
        """Tag the project's repo with the next version's tags and push to origin

        :param stage: the release stage to transition into (seal, send, sign)
        """
        with self.git.project_git(project_name) as git:
            final = git.get_final_tag()
            project_key = self._get_project_key(project_name)
            new_version = self.jira.get_pending_version_name(
                project_key,
                stage,
                git.get_rev_hash(ref="origin/develop")[:7],
                final.name,
                git.get_prerelease_tag(min_version=final).name,
            )
            git.tag_develop(tag_name=new_version)
            return new_version

    def _setup_repos(self):
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
