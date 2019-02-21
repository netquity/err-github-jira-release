import os
import shutil
import subprocess
import logging


class InvalidStageTransitionError(Exception):
    """An invalid stage (seal, sign, send) transition"""


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
def bump_version(release_type: str, stage: str, final_version: str, pre_version: str = None) -> str:
    """Perform a version bump in accordance with semver.org and PEP-440

    The versioning scheme is a combination of both Semantic Versioning and PEP-440 as described in the README

    :param release_type: the highest release type (of all merged tickets) since the `final_version`
    :param stage: the release stage to transition into (seal, send, sign)
    :param final_version: the latest final version name; the version to be upgraded (eg. 1.0.0)
    :param pre_version: the version containing information about the current release cycle:
                        pre-release version, rc count, stage (eg. 1.0.1-rc.5+sealed
                        it's possible that such a version does not yet exist, as when initiating the release sequence
                        for the first time
    :return: the new version string

    >>> bump_version('patch', 'seal', '1.0.0',)
    '1.0.1-rc.1+seal'
    >>> bump_version('minor', 'seal', '1.0.0',)
    '1.1.0-rc.1+seal'
    >>> bump_version('minor', 'send', '1.0.0', '1.1.0-rc.1+seal')
    '1.1.0-rc.1+send'

    # A full release cycle example
    >>> bump_version('major', 'seal', '1.0.0',)  # first seal
    '2.0.0-rc.1+seal'
    >>> bump_version('major', 'seal', '1.0.0',  '2.0.0-rc.1+seal',)  # second seal
    '2.0.0-rc.2+seal'
    >>> bump_version('major', 'send', '1.0.0', '2.0.0-rc.2+seal',)
    '2.0.0-rc.2+send'
    >>> bump_version('major', 'sign', '1.0.0', '2.0.0-rc.2+send',)
    '2.0.0-rc.2+sign'
    >>> bump_version('major', 'seal', '1.0.0', '2.0.0-rc.2+send',)
    '2.0.0-rc.3+seal'

    >>> bump_version('major', 'sign', '1.0.0', '2.0.0-rc.2+sign',)
    Traceback (most recent call last):
        ...
    helpers.InvalidStageTransitionError: Cannot transition from sign to sign
    """
    import semver
    if pre_version is not None:
        pre_stage = pre_version.split('+')[1]
        if not is_valid_transition(pre_stage, stage):
            raise InvalidStageTransitionError(f'Cannot transition from {pre_stage} to {stage}')
    verinfo = semver.parse(pre_version or final_version)
    bumped_semver = getattr(semver, f'bump_{release_type.lower()}')(final_version)

    if verinfo.get('prerelease', None):
        bumped_semver += f"-{verinfo['prerelease']}"
    if stage == 'seal':
        return semver.bump_prerelease(bumped_semver) + f"+{stage}"
    return bumped_semver + f"+{stage}"


def is_valid_transition(start: str, target: str) -> bool:
    """Check whether the stage transition is allowed

    The normal order of release stages is: seal, send, sign

    Some transitions are not allowed. For example, a release cannot be signed and then transitioned to sent. This
    function checks whether the given transition is one of the legal ones.

    :param start: the starting stage
    :param target: the desired end stage
    >>> is_valid_transition('sign', 'sign')
    False
    >>> is_valid_transition('sign', 'seal')
    False
    >>> is_valid_transition('sign', 'send')
    False
    >>> is_valid_transition('seal', 'send')
    True
    >>> is_valid_transition('send', 'sign')
    True
    >>> is_valid_transition('seal', 'seal')
    True
    >>> is_valid_transition('seal', 'seal')
    True
    >>> is_valid_transition('send', 'send')
    False
    >>> is_valid_transition('sign', 'sign')
    False
    >>> is_valid_transition('seal', 'sign')
    False
    >>> is_valid_transition('seal', 'send')
    True
    >>> is_valid_transition('send', 'seal')
    True
    >>> is_valid_transition('send', 'sign')
    True

    >>> is_valid_transition(None, 'seal')
    True
    >>> is_valid_transition(None, 'send')
    Traceback (most recent call last):
        ...
    ValueError: Can only transition a final to the sealed stage but got target=send
    >>> is_valid_transition(None, 'sign')
    Traceback (most recent call last):
        ...
    ValueError: Can only transition a final to the sealed stage but got target=sign
    >>> is_valid_transition('bad', 'seal')
    Traceback (most recent call last):
        ...
    ValueError: sign, seal, and send are the only valid stage options. Got start=bad, target=seal
    """
    if start is None:
        if target == 'seal':
            return True
        raise ValueError('Can only transition a final to the sealed stage but got target=%s' % target)
    if (
            start not in ['sign', 'seal', 'send']
            or target not in ['sign', 'seal', 'send']
    ):
        raise ValueError('sign, seal, and send are the only valid stage options. Got start=%s, target=%s' % (
            start, target
        ))
    return not any([
        start == 'sign',  # sign is the last stage; can't transition from it
        start == 'send' and target == 'send',
        start == 'seal' and target == 'sign',
    ])


def copytree(src: str, dst_parent: str, dst: str) -> str:
    """A simple wrapper for `shutil.copytree` which makes sure required parent directory exists

    :param src: the dir to be copied
    :param dst_parent: the parent location to copy the src to
    :param dst: the name of the folder to hold a copy of the src
    """
    if not os.path.exists(dst_parent):
        os.makedirs(dst_parent)  # creates intermediate dirs

    return shutil.copytree(src, f'{dst_parent}/{dst}')
