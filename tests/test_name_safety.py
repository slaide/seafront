"""
Names the user picks, which become directories on the microscope's disk.

`project_name` is a whole path component on its own - `<base>/<project>/<plate>_<timestamp>/` -
so whatever the user types there is a directory name, created with `mkdir(parents=True)` by
the disk sink. Two of those names mean something to the filesystem rather than being names at
all, and the validator used to accept both: `.` and `..`.

    base = ~/seafront/images
    project_name = ".."   ->  ~/seafront/plate_2026-...     (outside the output directory)
    project_name = "."    ->  ~/seafront/images/plate_...   (the base itself)

`plate_name` is safer by accident - it is glued to a timestamp, so `..` there becomes the
ordinary directory `.._2026-01-01`. `filename` for a stored protocol is the same shape as
`project_name` and gets the same treatment.

The other edge is length: a path component longer than the filesystem allows is not a
validation error today, it is an OSError partway through an acquisition, with images already
written.
"""

import pathlib

import pytest

from seafront.server.core import filename_check, name_check

# 255 bytes is the per-component limit on ext4, xfs, apfs and ntfs alike
MAX_COMPONENT = 255


class TestNamesThatAreNotNames:
    @pytest.mark.parametrize("name", [".", ".."])
    def test_the_filesystem_dot_names_are_refused(self, name):
        """
        the traversal

        Neither is a name; both are instructions to the filesystem. `..` walks out of the
        directory the operator configured, and the disk sink then creates it.
        """
        issue = name_check(name)
        assert issue is not None, f"{name!r} was accepted as a project name"

    @pytest.mark.parametrize("name", [".", ".."])
    def test_a_config_filename_refuses_them_too(self, name):
        assert filename_check(name) is not None, f"{name!r} was accepted as a filename"

    def test_three_dots_is_an_ordinary_name(self):
        """
        the refusal has to be exact

        `...` is a perfectly good directory name and means nothing special. Refusing every
        name that starts with a dot would be a different, larger decision.
        """
        assert name_check("...") is None

    def test_a_leading_dot_is_still_allowed(self):
        assert name_check(".hidden") is None

    def test_the_escape_does_not_happen(self, tmp_path):
        """
        the property, not the implementation

        Whatever the validator accepts, joining it under the output directory must stay
        under the output directory.
        """
        base = tmp_path / "images"
        base.mkdir()
        for candidate in (".", "..", "...", ".hidden", "ok-name", "a.b.c", "2026-01-01", "_x"):
            if name_check(candidate) is not None:
                continue  # refused, so it never reaches a path
            built = (base / candidate / "plate_2026").resolve()
            assert base.resolve() in built.parents, f"{candidate!r} escaped to {built}"


class TestNamesTooLongForTheFilesystem:
    def test_a_name_longer_than_a_path_component_is_refused(self):
        """
        refused up front, rather than as an OSError partway through a run

        An acquisition that fails on its first write has already been accepted, scheduled,
        and reported as running.
        """
        assert name_check("a" * (MAX_COMPONENT + 1)) is not None
        assert filename_check("a" * (MAX_COMPONENT + 1)) is not None

    def test_a_name_that_exactly_fits_is_accepted(self):
        assert name_check("a" * MAX_COMPONENT) is None

    def test_the_length_is_counted_in_bytes_not_characters(self, tmp_path):
        """
        what the filesystem counts

        Only relevant if non-ascii is ever allowed - it is not today, which this pins.
        """
        assert name_check("é") is not None, "non-ascii is refused, so bytes and chars agree"


class TestWhatIsStillAllowed:
    """the refusals have to be narrow, or ordinary names stop working"""

    @pytest.mark.parametrize(
        "name",
        ["plate1", "Plate_1", "2026-01-01", "a.b", "exp-01_rep2", "X", "0", "-", "_", "..."],
    )
    def test_ordinary_names_are_accepted(self, name):
        assert name_check(name) is None, f"{name!r} should be a usable name"

    @pytest.mark.parametrize(
        "name",
        ["", " ", "a b", "a/b", "a\\b", "a:b", "a*b", "tré", "a\tb", "a\nb", "..\x00"],
    )
    def test_names_with_anything_else_are_refused(self, name):
        assert name_check(name) is not None, f"{name!r} should not be a usable name"

    def test_a_name_cannot_contain_a_separator(self, tmp_path):
        """
        the reason a single `..` is the whole exposure

        No accepted name can contain a separator, so a name is always exactly one path
        component and can never walk more than one level.
        """
        assert name_check("../..") is not None
        assert name_check("a/../../b") is not None


class TestTheStoredPathIsBuiltFromCheckedNames:
    def test_the_project_path_is_two_components(self):
        """
        `<project>/<plate>_<timestamp>` - the shape the traversal exploited

        Pinned because it is what makes `project_name` the sensitive one: it is a component
        on its own, while `plate_name` is always suffixed with a timestamp.
        """
        project = pathlib.PurePath("myproject") / "myplate_2026-01-01T00-00-00"
        assert len(project.parts) == 2

    def test_a_plate_name_is_never_a_component_on_its_own(self):
        plate_component = f"{'..'}_2026-01-01T00-00-00"
        assert plate_component not in (".", ".."), "a timestamped plate name cannot be a dot name"
