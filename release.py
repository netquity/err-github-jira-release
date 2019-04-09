# coding: utf-8
"""An Errbot plugin for creating software releases by looking at completed work in Jira and GitHub."""
import errno
import logging
import os

from functools import partial
from typing import List, Mapping, Dict, Union, Optional

from errbot import BotPlugin, botcmd, ValidationException
from errbot.botplugin import recurse_check_structure
from jiraclient import JiraClient, NoJIRAIssuesFoundError
from errbot.backends.base import Message, Identifier
from gitclient import RepoManager, Repo, GitCommandError

import helpers

logger = logging.getLogger(os.path.basename(__file__))

try:
    from jira import JIRA, JIRAError
    from jira.resources import Version
    from github import Github
except ImportError:
    logger.error("Please install 'jira' and 'pygithub' Python packages")

MISSING_ERROR_MESSAGE = (
    'Error raised while trying to {stage} a release, but error '
    'key `{key}` does not exist in the `error_messages` dictionary.'
)


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
    error_messages = {
        'no_jira_issues': '{repo}: no <{issues_url}|Jira issues> found ({merge_summary}).',
        'invalid_transition': 'Invalid stage transition attempted when bumping {repo}.',
        'invalid_version': 'Invalid pre_version given when bumping {repo}.',
        'none_updated': '{active_stage}: no projects updated.',
        'mismatched_updates': (
            '{active_stage}: number of updated projects ({updated_projects}) '
            'does not match number of bumped projects ({bumped_counter}).'
        ),
        'unknown_git_error': '{repo}: unknown error encountered when performing git operations.'
    }

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
        self.git = RepoManager(self._get_git_config())  # pylint:disable=attribute-defined-outside-init

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

    @botcmd
    def seal(self, msg: Message, args) -> Optional[str]:  # pylint:disable=unused-argument
        """Initiate the release sequence by tagging updated projects"""
        return self._bump_repos_set(msg, helpers.Stages.SEALED)

    @botcmd
    def send(self, msg: Message, args) -> str:  # pylint:disable=unused-argument
        """Update SemVer metadata and tell the configured UAT channel that testing can begin

        This command is called only after the `seal` command is called and the sealed version set is tested and
        approved.
        """
        return self._bump_repos_set(msg, helpers.Stages.SENT)

    @botcmd
    def sign(self, msg: Message, args):  # pylint:disable=unused-argument
        from gitclient import GitCommandError
        updated_repos = self.git.get_repos_in_stage(helpers.Stages.SENT)
        for repo in updated_repos:
            key = self._get_project_key(repo)
            final = repo.get_final_tag()
            # FIXME: catch InvalidVersionNameError and InvalidStageTransitionError
            new_version_name = self.jira.get_pending_version_name(
                key,
                helpers.Stages.SIGNED,
                repo.get_rev_hash(ref="origin/develop")[:7],
                final.name,
                getattr(repo.get_prerelease_tag(min_version=final), 'name', None),
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
            # TODO: add release notes to changelog file
            # FIXME: catch IssueMergesCountMismatchError
            release_notes = self.jira.get_release_notes(jira_version, repo.get_merge_logs())
            is_hotfix = False  # TODO: TEMPORARY SHIM; REMOVE!!!
            try:
                if is_hotfix:
                    # TODO: need better exception handling
                    repo.add_release_notes_to_develop(new_version_name, release_notes)
                else:
                    repo.merge_and_create_release_commit(new_version_name, release_notes)
            except GitCommandError:
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
            Release._create_final_release(new_version_name, jira_version, repo, release_notes, is_hotfix)

            self._send_version_card(
                self.build_identifier(self.config['UAT_CHANNEL_IDENTIFIER']),
                repo=repo,
                card_dict=dict(
                    self._get_version_card(repo),
                    **{'New Version Name': new_version_name}
                ),
            )

    @staticmethod
    def _create_final_release(
            new_version_name: str,
            jira_version: Version,
            repo: Repo,
            release_notes: str,
            is_hotfix: bool = False,
    ) -> None:
        new_version_name = helpers.change_sha(new_version_name, repo.ref[:7])
        JiraClient.change_version_name(jira_version, new_version_name)
        repo.create_tag(tag_name=new_version_name)
        repo.create_ref(version_name=new_version_name)
        repo.create_release(release_notes=release_notes, version_name=new_version_name)
        if not is_hotfix:
            repo.update_develop()

    def _bump_repos_set(self, msg: Message, stage: helpers.Stages) -> Optional[str]:
        """Transition the entire set of updated projects to the given stage

        This method produces side-effects on Git (local and origin), and whatever chat backend is being used:
        - git: create tag and ref
        - backend: send cards and messages
        """
        # TODO: need to propagate errors and revert all changes if anything fails
        fail = partial(self._fail, identifier=msg.frm, stage=stage)

        if stage not in [helpers.Stages.SEALED, helpers.Stages.SENT]:
            raise ValueError('Given stage=%s not supported.' % stage)

        # For the first stage, we need all repos that have commits since the last final
        if stage == helpers.Stages.SEALED:
            updated_repos = self.git.get_updated_repos(not stage == helpers.Stages.SEALED)
        else:
            # If *not* at the beginning of a release cycle, we need all repos that were successfully transitioned
            # into the most recent stage. i.e. if we're ready to sign/finalize a release, we do that with all the
            # repos that were sent (which is a stage name) to UAT
            updated_repos = self.git.get_repos_in_stage(helpers.Stages(stage.value - 1))

        bumped_counter = 0
        for repo in updated_repos:
            # TODO: wrap in a try/except and roll back repos and jira on any kind of failure
            # TODO: these bumps can all be done asynchronously, they don't depend on each other
            try:
                release_type = self.jira.get_release_type(self._get_project_key(repo))
                new_version = self._bump_repo_tags(repo, stage)
                # CAUTION: Slack STRONGLY warns against sending more than 20 cards at a time:
                # https://api.slack.com/docs/message-attachments#attachment_limits
                self._send_version_card(
                    msg.frm if stage == helpers.Stages.SEALED
                    else self.build_identifier(self.config['UAT_CHANNEL_IDENTIFIER']),
                    repo=repo,
                    card_dict=dict(
                        self._get_version_card(repo),
                        **{'New Version Name': new_version}
                    ),
                )
                bumped_counter += 1
            except NoJIRAIssuesFoundError:
                fail(
                    'no_jira_issues',
                    repo=repo.name,
                    issues_url=self.jira.get_latest_issues_url(self._get_project_key(repo)),
                    merge_summary=self._get_merge_summary(repo),
                )
            except helpers.InvalidStageTransitionError:
                fail('invalid_transition', repo=repo.name)
            except helpers.InvalidVersionNameError:
                fail('invalid_version', repo=repo.name)
            except GitCommandError:
                return fail('unknown_git_error', repo=repo.name)

        if not bumped_counter > 0 and updated_repos:
            return fail('none_updated', active_stage=stage.verb)
        if bumped_counter != len(updated_repos):
            return fail(
                'mismatched_updates',
                active_stage=stage.verb,
                updated_repos=len(updated_repos),
                bumped_counter=bumped_counter,
            )
        # FIXME: doesn't work as a yield for `send` because need to send to different channels
        # yield f"{len(card_dict)} repos updated: \n\t• " + '\n\t• '.join(list(card_dict))

        return f'{bumped_counter} / {len(self._get_project_names())} repos updated since last release.'
        # return "I have sent your sealed version set to the UAT channel. Awaiting their approval."

    def _fail(self, key: str, identifier: Identifier, stage: helpers.Stages, **kwargs) -> None:
        """A helper method that simply sends an error message and logs it"""
        self.log.debug('Entering _fail: key=%s, stage=%s', key, stage)
        import sys
        if sys.exc_info()[0] is not None:
            self.log.exception('An exception occurred while performing a release')

        try:
            msg = self.error_messages[key]
        except KeyError:
            self.log.error('Unknown error raised when release stage=%s, key=%s', stage, key)
            msg = MISSING_ERROR_MESSAGE.format(stage=stage, key=key)
        message_string = msg.format(**kwargs)
        self.log.warning(message_string)
        return self.send_card(to=identifier, body=message_string, color='red',)

    def _get_project_names(self) -> List[str]:
        """Get the list of project names from the configuration"""
        return list(self.config['projects'])

    def _get_project_key(self, repo: Repo) -> str:
        """Get the Jira project key for the given repo name"""
        # TODO: catch `KeyError`
        return self.config['projects'][repo.name]

    def _get_jira_config(self) -> dict:
        """Return data required for initializing JiraClient"""
        return {
            'URL': self.config['JIRA_URL'],
            'USER': self.config['JIRA_USER'],
            'PASS': self.config['JIRA_PASS'],
            'TEMPLATE_DIR': self.config['TEMPLATE_DIR'],
        }

    def _get_git_config(self) -> dict:
        """Return data required for initializing RepoManager"""
        return {
            'REPOS_ROOT': self.config['REPOS_ROOT'],
            'GITHUB_TOKEN': self.config['GITHUB_TOKEN'],
            'PROJECT_NAMES': self._get_project_names(),
        }

    def _get_version_card(self, repo: Repo, release_type: str) -> Dict:
        self.log.debug('%s: getting version card', repo.name)
        final = repo.get_final_tag()
        project_key = self._get_project_key(repo)
        return {
            'Key': project_key,
            'Release Type': release_type,

            'Previous Version': f'<{final.url}|{final.name}>',
            'Previous vCommit': final.sha,

            'Merge Count': Release._get_merge_summary(repo),
            # TODO: it would be nice to be able to dynamically pass in functions for fields to show up on the card
            'New Migrations': repo.get_migration_count(),  # FIXME: too django-specific

            # To be removed for `fields`
            'GitHub Tag Comparison': repo.get_compare_url(),
            # TODO: find a good public source for thumbnails; follow license
            'thumbnail': 'https://static.thenounproject.com/png/1662598-200.png',
        }

    def _send_version_card(
            self,
            identifier: Identifier,
            repo: Repo,
            card_dict: Dict[str, Union[str, int]],
    ) -> None:
        """Send the Slack card containing version set information

        :param message:
        :param repo:
        :param card_dict: a dict of values to be displayed on the version card
        """
        self.send_card(  # CAUTION: Slack STRONGLY warns against sending more than 20 cards at a time
            title=f'{repo.name} - {card_dict.pop("New Version Name")}',
            link=card_dict.pop('GitHub Tag Comparison'),
            to=identifier,
            thumbnail=card_dict.pop('thumbnail'),
            fields=tuple(card_dict.items()),
            color='green',
        )
        self.log.debug('%s sent version card', repo.name)

    def _bump_repo_tags(self, repo: Repo, stage: str) -> str:
        """Tag the repo's repo with the next version's tags and push to origin

        :param stage: the release stage to transition into (seal, send, sign)
        """
        final = repo.get_final_tag()
        project_key = self._get_project_key(repo)
        new_version = self.jira.get_pending_version_name(
            project_key,
            stage,
            repo.get_rev_hash(ref="origin/develop")[:7],
            final.name,
            getattr(repo.get_prerelease_tag(min_version=final), 'name', None),
        )
        repo.tag_develop(tag_name=new_version)
        return new_version

    def _setup_repos(self):
        """Clone the projects in the configuration into the `REPOS_ROOT` if they do not exist already."""
        try:
            os.makedirs(self.config['REPOS_ROOT'])
        except OSError as exc:
            # If the error is that the directory already exists, we don't care about it
            if exc.errno != errno.EEXIST:
                raise exc

        for project in self.config['projects']:
            if not os.path.exists(os.path.join(self.config['REPOS_ROOT'], project)):
                # Possible race condition if folder somehow gets created between check and creation
                helpers.run_subprocess(
                    ['git', 'clone', f"git@github.com:{project}.git", project],
                    cwd=self.config['REPOS_ROOT'],
                )

    @staticmethod
    def _get_merge_summary(repo: Repo) -> str:
        """Return a link to GitHub's issue search showing the merged PRs """
        return '<{url}|{pr_count} merged PR(s)>'.format(
            url=repo.get_merged_prs_url(),
            pr_count=repo.get_merge_count(),
        )
