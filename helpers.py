import os
import shutil
import subprocess
import logging
from enum import Enum

from errbot import ValidationException

logger = logging.getLogger(__file__)


class Stages(Enum):
    """The different release stages"""
    SEALED = 1
    SENT = 2
    SIGNED = 3

    @property
    def verb(self):
        """Provide the verb used to transition into the given stage

        For example, to transition into the SEALED stage, the user needs to perform the `seal` action"""
        if self.name == Stages.SEALED.name:
            return 'seal'
        if self.name == Stages.SENT.name:
            return 'send'
        return 'sign'


S = Stages  # Shortcuts  pylint:disable=invalid-name


class InvalidStageTransitionError(ValidationException):
    """An invalid stage (seal, sign, send) transition"""


class InvalidVersionNameError(ValidationException):
    """Given version string is not valid"""


def update_changelog_file(
        changelog_path: str,
        release_notes: str,
        log: logging.Logger
):
    """Prepend the given release notes to CHANGELOG.md."""
    # TODO: exceptions
    try:
        with open(changelog_path, 'r') as changelog:
            original_contents = changelog.read()
        with open(changelog_path, 'w') as changelog:
            changelog.write(release_notes + "\n" + original_contents)
    except OSError as exc:
        log.exception('An unknown error occurred while updating the changelog file.')
        raise exc


def run_subprocess(args: list, cwd: str = None) -> subprocess.CompletedProcess:
    """Run the local command described by `args` with some defaults applied."""
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine out/err into stdout; stderr will be None
        universal_newlines=True,
        check=True,
        cwd=cwd,
    )


# TODO: add hotfix support
def bump_version(release_type: str, stage: str, sha: str, final_version: str, pre_version: str = None) -> str:
    """Perform a version bump in accordance with semver.org and PEP-440

    The versioning scheme is a combination of both Semantic Versioning and PEP-440 as described in the README

    :param release_type: the highest release type (of all merged tickets) since the `final_version`
    :param stage: the release stage to transition into (seal, send, sign)
    :param sha: short commit hash to append to metadata segment, usually origin/develop's HEAD
    :param final_version: the latest final version name; the version to be upgraded (eg. 1.0.0)
    :param pre_version: the version containing information about the current release cycle:
                        prerelease version, rc count, stage, sha (eg. 1.0.1-rc.5+sealed.666d074)
                        it's possible that such a version does not yet exist, as when initiating the release sequence
                        for the first time
    :return: the new version string WITHOUT the `v` prefix

    >>> bump_version('patch', S.SEALED.verb, 'cafe7f7', '1.0.0+666d070',)
    '1.0.1-rc.1+seal.cafe7f7'
    >>> bump_version('minor', S.SEALED.verb, 'cafe7f7', '1.0.0+666d071',)
    '1.1.0-rc.1+seal.cafe7f7'
    >>> bump_version('minor', S.SENT.verb, 'cafe7f7', '1.0.0', '1.1.0-rc.1+seal.666d072')
    '1.1.0-rc.1+send.cafe7f7'

    # A full release cycle example
    >>> bump_version('major', S.SEALED.verb, 'cafe7f7', '1.0.0+666d073',)  # first seal
    '2.0.0-rc.1+seal.cafe7f7'
    >>> bump_version('major', S.SEALED.verb, 'cafe7f7', '1.0.0',  '2.0.0-rc.1+seal.666d074',)  # second seal
    '2.0.0-rc.2+seal.cafe7f7'
    >>> bump_version('major', S.SENT.verb, 'cafe7f7', '1.0.0', '2.0.0-rc.2+seal.666d075',)
    '2.0.0-rc.2+send.cafe7f7'
    >>> bump_version('major', S.SIGNED.verb, 'cafe7f7', '1.0.0', '2.0.0-rc.2+send.666d076',)
    '2.0.0+cafe7f7'
    >>> bump_version('major', S.SEALED.verb, 'cafe7f7', '1.0.0', '2.0.0-rc.2+send.666d077',)
    '2.0.0-rc.3+seal.cafe7f7'

    >>> bump_version('major', S.SIGNED.verb, 'cafe7f7', '1.0.0', '2.0.0-rc.2+sign.666d078',)
    Traceback (most recent call last):
        ...
    helpers.InvalidStageTransitionError: Cannot transition from sign to sign
    >>> bump_version('major', S.SEALED.verb, 'cafe7f7', '1.0.0', '2.0.0-rc.2-send.666d077',)  # + → -
    Traceback (most recent call last):
        ...
    helpers.InvalidVersionNameError: 2.0.0-rc.2-send.666d077 is not a valid version string
    """
    import semver
    if pre_version is not None:
        try:
            pre_stage = pre_version.split('+')[1].split('.')[0]  # get the name of the stage
        except IndexError:
            raise InvalidVersionNameError(f'{pre_version} is not a valid version string')
        if not is_valid_transition(pre_stage, stage):
            raise InvalidStageTransitionError(f'Cannot transition from {pre_stage} to {stage}')
    verinfo = semver.parse(pre_version or final_version)
    bumped_semver = getattr(semver, f'bump_{release_type.lower()}')(final_version)

    if stage == S.SIGNED.verb:
        return bumped_semver + f"+{sha}"
    if verinfo.get('prerelease', None):
        bumped_semver += f"-{verinfo['prerelease']}"
    if stage == S.SEALED.verb:
        return semver.bump_prerelease(bumped_semver) + f"+{stage}.{sha}"
    return bumped_semver + f"+{stage}.{sha}"


def is_valid_transition(start: str, target: str) -> bool:
    """Check whether the stage transition is allowed

    The normal order of release stages is: seal, send, sign

    Some transitions are not allowed. For example, a release cannot be signed and then transitioned to sent. This
    function checks whether the given transition is one of the legal ones.

    :param start: the starting stage
    :param target: the desired end stage
    >>> is_valid_transition(S.SIGNED.verb, S.SIGNED.verb)
    False
    >>> is_valid_transition(S.SIGNED.verb, S.SEALED.verb)
    False
    >>> is_valid_transition(S.SIGNED.verb, S.SENT.verb)
    False
    >>> is_valid_transition(S.SEALED.verb, S.SENT.verb)
    True
    >>> is_valid_transition(S.SENT.verb, S.SIGNED.verb)
    True
    >>> is_valid_transition(S.SEALED.verb, S.SEALED.verb)
    True
    >>> is_valid_transition(S.SEALED.verb, S.SEALED.verb)
    True
    >>> is_valid_transition(S.SENT.verb, S.SENT.verb)
    False
    >>> is_valid_transition(S.SIGNED.verb, S.SIGNED.verb)
    False
    >>> is_valid_transition(S.SEALED.verb, S.SIGNED.verb)
    False
    >>> is_valid_transition(S.SEALED.verb, S.SENT.verb)
    True
    >>> is_valid_transition(S.SENT.verb, S.SEALED.verb)
    True
    >>> is_valid_transition(S.SENT.verb, S.SIGNED.verb)
    True

    >>> is_valid_transition(None, S.SEALED.verb)
    True
    >>> is_valid_transition(None, S.SENT.verb)
    Traceback (most recent call last):
        ...
    ValueError: Can only transition a final to the sealed stage but got target=send
    >>> is_valid_transition(None, S.SIGNED.verb)
    Traceback (most recent call last):
        ...
    ValueError: Can only transition a final to the sealed stage but got target=sign
    >>> is_valid_transition('bad', S.SEALED.verb)
    Traceback (most recent call last):
        ...
    ValueError: sign, seal, and send are the only valid stage options. Got start=bad, target=seal
    """
    if start is None:
        if target == S.SEALED.verb:
            return True
        raise ValueError('Can only transition a final to the sealed stage but got target=%s' % target)
    if (
            start not in [S.SIGNED.verb, S.SEALED.verb, S.SENT.verb]
            or target not in [S.SIGNED.verb, S.SEALED.verb, S.SENT.verb]
    ):
        raise ValueError('sign, seal, and send are the only valid stage options. Got start=%s, target=%s' % (
            start, target
        ))
    return not any([
        start == S.SIGNED.verb,  # sign is the last stage; can't transition from it
        start == S.SENT.verb and target == S.SENT.verb,
        start == S.SEALED.verb and target == S.SIGNED.verb,
    ])


def copytree(src: str, dst_parent: str, dst: str) -> str:
    """A simple wrapper for `shutil.copytree` which makes sure required parent directory exists

    :param src: the dir to be copied
    :param dst_parent: the parent location to copy the src to
    :param dst: the name of the folder to hold a copy of the src
    """
    if not os.path.exists(dst_parent):
        os.makedirs(dst_parent)  # creates intermediate dirs

    try:
        return shutil.copytree(src, f'{dst_parent}/{dst}')
    except FileExistsError:
        logger.warning('Git repo backup %s already exists; ignoring.', f'{dst_parent}/{dst}')
