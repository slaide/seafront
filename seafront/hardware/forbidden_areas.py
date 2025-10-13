"""
Forbidden areas management for microscope stage movement safety.

This module provides AABB (Axis-Aligned Bounding Box) based forbidden area checking
to prevent the microscope stage from moving to physically dangerous locations.
"""

import json

import json5
from pydantic import BaseModel, Field, validator


class ForbiddenArea(BaseModel):
    """
    Represents a rectangular forbidden area in physical coordinates (mm).

    Uses AABB (Axis-Aligned Bounding Box) representation for fast intersection checking.
    """

    name: str = Field(..., description="Human-readable name for this forbidden area")
    min_x_mm: float = Field(..., description="Minimum X coordinate (left edge) in mm")
    max_x_mm: float = Field(..., description="Maximum X coordinate (right edge) in mm")
    min_y_mm: float = Field(..., description="Minimum Y coordinate (bottom edge) in mm")
    max_y_mm: float = Field(..., description="Maximum Y coordinate (top edge) in mm")
    reason: str = Field(default="", description="Optional reason why this area is forbidden")

    @validator('max_x_mm')
    def validate_x_bounds(cls, v, values):
        """Ensure max_x_mm >= min_x_mm"""
        if 'min_x_mm' in values and v < values['min_x_mm']:
            raise ValueError(f"max_x_mm ({v}) must be >= min_x_mm ({values['min_x_mm']})")
        return v

    @validator('max_y_mm')
    def validate_y_bounds(cls, v, values):
        """Ensure max_y_mm >= min_y_mm"""
        if 'min_y_mm' in values and v < values['min_y_mm']:
            raise ValueError(f"max_y_mm ({v}) must be >= min_y_mm ({values['min_y_mm']})")
        return v

    def contains_point(self, x_mm: float, y_mm: float) -> bool:
        """
        Check if a point is inside this forbidden area.

        Args:
            x_mm: X coordinate in mm
            y_mm: Y coordinate in mm

        Returns:
            True if the point is inside this forbidden area
        """
        return (self.min_x_mm <= x_mm <= self.max_x_mm and
                self.min_y_mm <= y_mm <= self.max_y_mm)

    def intersects_circle(self, center_x_mm: float, center_y_mm: float, radius_mm: float) -> bool:
        """
        Check if a circular area intersects with this forbidden area.

        Useful for checking if movement destinations near forbidden areas are safe,
        accounting for objective size or safety margins.

        Args:
            center_x_mm: Circle center X coordinate in mm
            center_y_mm: Circle center Y coordinate in mm
            radius_mm: Circle radius in mm

        Returns:
            True if the circle intersects this forbidden area
        """
        # Find the closest point on the AABB to the circle center
        closest_x = max(self.min_x_mm, min(center_x_mm, self.max_x_mm))
        closest_y = max(self.min_y_mm, min(center_y_mm, self.max_y_mm))

        # Calculate distance from circle center to closest point
        distance_squared = (center_x_mm - closest_x) ** 2 + (center_y_mm - closest_y) ** 2

        return distance_squared <= radius_mm ** 2

    def __str__(self) -> str:
        return f"ForbiddenArea('{self.name}': [{self.min_x_mm}, {self.max_x_mm}] x [{self.min_y_mm}, {self.max_y_mm}])"


class ForbiddenAreaList(BaseModel):
    """
    Manages a collection of forbidden areas with efficient checking operations.
    """

    areas: list[ForbiddenArea] = Field(default_factory=list, description="List of forbidden areas")

    def add_area(self, area: ForbiddenArea) -> None:
        """Add a forbidden area to the list."""
        self.areas.append(area)

    def is_position_forbidden(self, x_mm: float, y_mm: float) -> tuple[bool, ForbiddenArea | None]:
        """
        Check if a position is in any forbidden area.

        Args:
            x_mm: X coordinate in mm
            y_mm: Y coordinate in mm

        Returns:
            Tuple of (is_forbidden, forbidden_area). forbidden_area is None if position is allowed.
        """
        for area in self.areas:
            if area.contains_point(x_mm, y_mm):
                return True, area
        return False, None

    def is_movement_safe(self, x_mm: float, y_mm: float, safety_radius_mm: float = 0.0) -> tuple[bool, ForbiddenArea | None]:
        """
        Check if movement to a position is safe, considering safety margins.

        Args:
            x_mm: Target X coordinate in mm
            y_mm: Target Y coordinate in mm
            safety_radius_mm: Safety margin radius around the target position

        Returns:
            Tuple of (is_safe, conflicting_area). conflicting_area is None if movement is safe.
        """
        if safety_radius_mm <= 0.0:
            return not self.is_position_forbidden(x_mm, y_mm)[0], self.is_position_forbidden(x_mm, y_mm)[1]

        for area in self.areas:
            if area.intersects_circle(x_mm, y_mm, safety_radius_mm):
                return False, area
        return True, None

    def get_conflicting_areas(self, x_mm: float, y_mm: float, safety_radius_mm: float = 0.0) -> list[ForbiddenArea]:
        """
        Get all forbidden areas that would conflict with a movement.

        Args:
            x_mm: Target X coordinate in mm
            y_mm: Target Y coordinate in mm
            safety_radius_mm: Safety margin radius around the target position

        Returns:
            List of conflicting forbidden areas
        """
        conflicting = []
        for area in self.areas:
            if safety_radius_mm <= 0.0:
                if area.contains_point(x_mm, y_mm):
                    conflicting.append(area)
            else:
                if area.intersects_circle(x_mm, y_mm, safety_radius_mm):
                    conflicting.append(area)
        return conflicting

    @classmethod
    def from_json_string(cls, json_str: str) -> "ForbiddenAreaList":
        """
        Create ForbiddenAreaList from JSON string.

        Expected format (array of forbidden area objects):
        [
          {
            "name": "Top-left corner",
            "min_x_mm": 0.0,
            "max_x_mm": 10.0,
            "min_y_mm": 60.0,
            "max_y_mm": 70.0,
            "reason": "Stage mechanical limit"
          }
        ]

        Args:
            json_str: JSON string containing array of forbidden areas

        Returns:
            ForbiddenAreaList instance

        Raises:
            ValueError: If JSON is invalid or areas are malformed
        """
        try:
            data = json5.loads(json_str)
            if not isinstance(data, list):
                raise ValueError("JSON must be an array of forbidden area objects")

            areas = []
            for i, area_data in enumerate(data):
                try:
                    area = ForbiddenArea(**area_data)
                    areas.append(area)
                except Exception as e:
                    raise ValueError(f"Invalid forbidden area at index {i}: {e}") from e

            return cls(areas=areas)

        except ValueError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

    def to_json_string(self) -> str:
        """
        Convert ForbiddenAreaList to JSON string.

        Returns:
            JSON string representation (array of forbidden area objects)
        """
        return json.dumps([area.dict() for area in self.areas], indent=2)

    def __len__(self) -> int:
        return len(self.areas)

    def __iter__(self):
        return iter(self.areas)

    def __str__(self) -> str:
        return f"ForbiddenAreaList({len(self.areas)} areas)"


def create_default_forbidden_areas() -> ForbiddenAreaList:
    """
    Create default forbidden areas for typical SQUID microscope setup.

    These represent common problematic areas like plate corners and mechanical limits.

    Returns:
        ForbiddenAreaList with default areas
    """
    default_areas = [
        # Plate corners that might cause mechanical issues
        ForbiddenArea(
            name="Top-left plate corner",
            min_x_mm=0.0,
            max_x_mm=8.0,
            min_y_mm=77.0,
            max_y_mm=85.0,
            reason="Plate holder interference risk"
        ),
        ForbiddenArea(
            name="Top-right plate corner",
            min_x_mm=119.0,
            max_x_mm=127.0,
            min_y_mm=77.0,
            max_y_mm=85.0,
            reason="Plate holder interference risk"
        ),
        ForbiddenArea(
            name="Bottom-left plate corner",
            min_x_mm=0.0,
            max_x_mm=8.0,
            min_y_mm=0.0,
            max_y_mm=8.0,
            reason="Plate holder interference risk"
        ),
        ForbiddenArea(
            name="Bottom-right plate corner",
            min_x_mm=119.0,
            max_x_mm=127.0,
            min_y_mm=0.0,
            max_y_mm=8.0,
            reason="Plate holder interference risk"
        ),
    ]

    return ForbiddenAreaList(areas=default_areas)
