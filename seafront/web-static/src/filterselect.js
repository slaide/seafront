"use strict";

/**
 * Register filter select element with validation.
 * Shows error state when filter wheel is available, channel is enabled, but filter is unspecified.
 *
 * @param {HTMLSelectElement} el - The select element to validate
 * @param {() => object} getChannel - Function that returns the current channel object
 * @param {() => boolean} getIsFilterWheelAvailable - Function that returns if filter wheel is available
 */
export function registerFilterSelect(el, getChannel, getIsFilterWheelAvailable) {
    if (el.tagName !== 'SELECT') {
        console.warn('registerFilterSelect called on non-select element:', el);
        return;
    }

    /**
     * Validate the filter select and show/hide error feedback
     */
    function validateFilter() {
        const channel = getChannel();
        const isFilterWheelAvailable = getIsFilterWheelAvailable();

        let isValid = true;
        let errorMessage = "";

        // Remove previous error styling
        el.classList.remove("filter-select-error");

        // Remove any existing tooltips for this element
        const existingTooltips = document.querySelectorAll('.filter-select-tooltip');
        existingTooltips.forEach(tooltip => {
            if (tooltip._associatedElement === el) {
                tooltip.remove();
            }
        });

        // If filter wheel is available and channel is enabled
        if (isFilterWheelAvailable && channel.enabled) {
            // Check if filter is unspecified (null, undefined, or '__UI_NONE__')
            if (!channel.filter_handle || channel.filter_handle === '__UI_NONE__') {
                isValid = false;
                errorMessage = "Filter selection is required when filter wheel is available";
            }
        }

        if (!isValid) {
            // Add error styling
            el.classList.add("filter-select-error");

            // Create tooltip
            const tooltip = document.createElement('div');
            tooltip.className = 'number-input-tooltip'; // Reuse number-input-tooltip styling
            tooltip.textContent = errorMessage;
            tooltip._associatedElement = el; // Mark which element this tooltip is for

            // Position tooltip within viewport
            const rect = el.getBoundingClientRect();
            document.body.appendChild(tooltip);

            // Get tooltip dimensions after adding to DOM
            const tooltipRect = tooltip.getBoundingClientRect();

            // Calculate optimal position
            let left = rect.left;
            let top = rect.top - tooltipRect.height - 8; // 8px gap above select
            let isAbove = true;

            // If tooltip would go above viewport, show below instead
            if (top < 10) { // 10px margin from top
                top = rect.bottom + 8; // 8px gap below select
                isAbove = false;
            }

            // If still below viewport, constrain to bottom
            const maxTop = window.innerHeight - tooltipRect.height - 10; // 10px margin
            if (top > maxTop && !isAbove) {
                top = maxTop;
            }

            // Constrain horizontally to viewport
            const maxLeft = window.innerWidth - tooltipRect.width - 10; // 10px margin
            if (left > maxLeft) {
                left = maxLeft;
            }
            if (left < 10) { // 10px margin from left edge
                left = 10;
            }

            // Calculate arrow position relative to element center
            const elementCenter = rect.left + rect.width / 2;
            let arrowLeft = elementCenter - left;

            // Constrain arrow position within tooltip bounds
            const minArrowLeft = 10;
            const maxArrowLeft = tooltipRect.width - 20;
            if (arrowLeft < minArrowLeft) arrowLeft = minArrowLeft;
            if (arrowLeft > maxArrowLeft) arrowLeft = maxArrowLeft;

            // Apply position and styling
            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
            tooltip.classList.add(isAbove ? 'tooltip-above' : 'tooltip-below');
            tooltip.style.setProperty('--arrow-left', arrowLeft + 'px');
        }
    }

    // Validate on change event
    el.addEventListener("change", validateFilter);

    // Validate on blur event
    el.addEventListener("blur", validateFilter);

    // Trigger initial validation
    requestAnimationFrame(() => {
        validateFilter();
    });
}
