"""
Forbidden areas management for microscope stage movement safety.

This module provides AABB (Axis-Aligned Bounding Box) based forbidden area checking
to prevent the microscope stage from moving to physically dangerous locations.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Self


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

    @model_validator(mode='after')
    def validate_bounds(self) -> Self:
        """Ensure max coordinates >= min coordinates"""
        if self.max_x_mm < self.min_x_mm:
            raise ValueError(f"max_x_mm ({self.max_x_mm}) must be >= min_x_mm ({self.min_x_mm})")
        if self.max_y_mm < self.min_y_mm:
            raise ValueError(f"max_y_mm ({self.max_y_mm}) must be >= min_y_mm ({self.min_y_mm})")
        return self

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

    def is_movement_safe(self, x_mm: float, y_mm: float) -> tuple[bool, ForbiddenArea | None]:
        """
        Check if movement to a position is safe.

        Args:
            x_mm: Target X coordinate in mm
            y_mm: Target Y coordinate in mm

        Returns:
            Tuple of (is_safe, conflicting_area). conflicting_area is None if movement is safe.
        """
        return not self.is_position_forbidden(x_mm, y_mm)[0], self.is_position_forbidden(x_mm, y_mm)[1]

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
