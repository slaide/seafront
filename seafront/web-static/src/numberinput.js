"use strict";

/**
 * get number of decimal digits for some number n
 * (max 9 decimal digits)
 * @param {null|number|string} n
 * @returns {number} number of decimal digits
 */
function getnumberofdecimaldigits(n) {
    if (n == null) return 0;

    // str->float->str round trip to truncate rounding errors in string conversion
    const numberAsStringComponents = parseFloat(parseFloat(n + "").toFixed(9))
        .toString()
        .split(".");
    if (numberAsStringComponents.length > 1) {
        return numberAsStringComponents[1].length;
    }
    return 0;
}

class NumberInput extends HTMLInputElement {
    get numvalue() {
        return parseFloat(super.value);
    }
}
window.customElements.define("number-input", NumberInput, { extends: "input" });

/**
 * @typedef {{
 *   minvalue: number | null;
 *   maxvalue: number | null;
 *   stepvalue: number | null;
 * }} NumberLimits
 */

/**
 * register input event overrides on (functionally) number input element.
 * enforces numberic limits and step size while typing, so that the field may never contain invalid values.
 * @param {NumberInput} el
 */
export function registerNumberInput(el) {
    // ensure element is supposed to be a number input
    if (el.type !== "number") {
        console.warn(
            `called registerNumberInput on element with type=${el.type}`,
            el,
        );
        return;
    }

    if (el.getAttribute("is") !== "number-input") {
        console.warn(
            "number input registered must have attribute is='number-input' for x-model to work.",
            el,
        );
        return;
    }

    // change its type to text input, because a browser with type=number will not
    // allow certain operations on the element input.
    el.setAttribute("type", "text");

    /**
     * Get current resolved values from attributes
     * This evaluates attribute expressions against the global Alpine data
     * @returns {NumberLimits}
     */
    function getCurrentLimits() {
        const currentMin = el.getAttribute('min');
        const currentMax = el.getAttribute('max'); 
        const currentStep = el.getAttribute('step');
        
        // Try to evaluate expressions like "limits.imaging_exposure_time_ms.min"
        let minvalue = null;
        let maxvalue = null;
        let stepvalue = null;
        
        minvalue = parseFloat(currentMin||"");
        if (isNaN(minvalue)) minvalue = null;
        
        maxvalue = parseFloat(currentMax||"");
        if (isNaN(maxvalue)) maxvalue = null;
        
        stepvalue = parseFloat(currentStep||"");
        if (isNaN(stepvalue)) stepvalue = null;
        
        return { minvalue, maxvalue, stepvalue };
    }

    // Note: numdigits will be calculated dynamically in validation
    /**
     *
     * @param {number} newval
     * @param {NumberLimits?} limits - optional limits object with minvalue, maxvalue, stepvalue
     * @returns {string}
     */
    function formatNumberInput(newval, limits = null) {
        const { minvalue: minVal, maxvalue: maxVal, stepvalue: stepVal } = limits || getCurrentLimits();
        const numdigits = getnumberofdecimaldigits(stepVal);
        
        if (!isFinite(newval)) {
            return (el.value || minVal || maxVal || 0) + "";
        }
        if (maxVal != null && newval > maxVal) {
            return maxVal.toFixed(numdigits);
        }
        if (minVal != null && newval < minVal) {
            return minVal.toFixed(numdigits);
        }
        if (getnumberofdecimaldigits(newval) > numdigits) {
            return newval.toFixed(numdigits);
        }

        return parseFloat(newval + "").toFixed(numdigits);
    }

    // Set up initial sizing and placeholder
    const initialLimits = getCurrentLimits();
    try {
        const maxnumchars = Math.max(
            formatNumberInput(initialLimits.minvalue ?? 0).length,
            formatNumberInput(initialLimits.maxvalue ?? 0).length,
            formatNumberInput(el.numvalue).length,
        );
        if (maxnumchars != null && !isNaN(maxnumchars)) {
            el.setAttribute("size", `${maxnumchars}`);
        }
    } catch (e) {
        // Fallback sizing
        el.setAttribute("size", "10");
    }

    let placeholdervalue = parseFloat(el.getAttribute("placeholder") ?? "");
    if (!isNaN(placeholdervalue)) {
        try {
            el.setAttribute("placeholder", formatNumberInput(placeholdervalue));
        } catch (e) {
            // Keep original placeholder
        }
    }

    /**
     * Validate the input and show/hide error feedback
     * @param {boolean} forceValid - if true, correct invalid values instead of just showing error
     */
    function validateInput(forceValid = false) {
        /** @type {number} */
        const currentValue = el.numvalue;
        let isValid = true;
        let errorMessage = "";

        // Get current limits (try to resolve dynamic values)
        const limits = getCurrentLimits();
        
        const minValue = limits.minvalue;
        const maxValue = limits.maxvalue;
        const stepValue = limits.stepvalue;

        // Remove previous error styling
        el.classList.remove("number-input-error");
        // Remove any existing tooltips for this element
        const existingTooltips = document.querySelectorAll('.number-input-tooltip');
        existingTooltips.forEach(tooltip => tooltip.remove());

        if (el.value.trim() === "") {
            // Empty input validation - always invalid, never auto-fill
            isValid = false;
            errorMessage = "Cannot be empty";
            console.log('Invalid: empty input');
            // Note: We no longer auto-fill with placeholder on blur - force user to enter a value
        } else if (isNaN(currentValue)) {
            // Check if this is a valid intermediate state during typing
            const isIntermediateState = !forceValid && (
                el.value === '-' ||           // Just a minus sign
                el.value === '+' ||           // Just a plus sign  
                el.value === '.' ||           // Just a decimal point
                el.value === '-.' ||          // Minus and decimal
                el.value === '+.' ||          // Plus and decimal
                el.value.match(/^-?0+$/)      // "-0", "00", "-00", etc.
            );
            
            if (isIntermediateState) {
                // Allow intermediate states during typing, but not on blur
                isValid = true;
                console.log('Allowing intermediate state during typing:', el.value);
            } else {
                // Invalid number format (contains letters, multiple dots, etc.)
                isValid = false;
                if (el.value.match(/[a-zA-Z]/)) {
                    errorMessage = "Contains letters - only numbers allowed";
                } else if (el.value.match(/\.\./)) {
                    errorMessage = "Invalid number format - multiple decimal points";
                } else if (el.value.match(/[^0-9.\-\+e]/)) {
                    errorMessage = "Invalid characters - only numbers, dots, and minus allowed";
                } else {
                    errorMessage = "Invalid number format";
                }
                console.log('Invalid: not a number', el.value);
            }
        } else {
            // For valid numbers, check range and step constraints
            if (minValue != null && currentValue < minValue) {
                isValid = false;
                errorMessage = `Value too small - minimum is ${minValue}`;
            } else if (maxValue != null && currentValue > maxValue) {
                isValid = false;
                errorMessage = `Value too big - maximum is ${maxValue}`;
            } else if (stepValue != null && getnumberofdecimaldigits(currentValue) > getnumberofdecimaldigits(stepValue)) {
                const maxDecimals = getnumberofdecimaldigits(stepValue);
                isValid = false;
                errorMessage = `Too many decimal places - maximum ${maxDecimals} allowed`;
            }
        }

        if (!isValid) {
            // Add error styling regardless of forceValid
            el.classList.add("number-input-error");
            
            // Create tooltip
            const tooltip = document.createElement('div');
            tooltip.className = 'number-input-tooltip';
            tooltip.textContent = errorMessage;
            
            // Add range info if available
            let rangeInfo = [];
            if (minValue != null) rangeInfo.push(`Min: ${minValue}`);
            if (maxValue != null) rangeInfo.push(`Max: ${maxValue}`);
            if (stepValue != null) rangeInfo.push(`Step: ${stepValue}`);
            
            if (rangeInfo.length > 0) {
                tooltip.textContent += ` (${rangeInfo.join(', ')})`;
            }
            
            // Position tooltip within viewport
            const rect = el.getBoundingClientRect();
            document.body.appendChild(tooltip);
            
            // Get tooltip dimensions after adding to DOM
            const tooltipRect = tooltip.getBoundingClientRect();
            
            // Calculate optimal position
            let left = rect.left;
            let top = rect.top - tooltipRect.height - 8; // 8px gap above input
            let isAbove = true;
            
            // If tooltip would go above viewport, show below input instead
            if (top < 10) { // 10px margin from top
                top = rect.bottom + 8; // 8px gap below input
                isAbove = false;
            }
            
            // If still below viewport, constrain to bottom and keep above
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
            
            // Calculate arrow position relative to input center
            const inputCenter = rect.left + rect.width / 2;
            let arrowLeft = inputCenter - left;
            
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
        } else if (isValid && forceValid) {
            // Only format if the value is valid
            el.value = formatNumberInput(currentValue, { minvalue: minValue, maxvalue: maxValue, stepvalue: stepValue });
        }
        // Note: If value is invalid, we preserve the invalid value and keep the error styling visible
    }

    el.addEventListener("blur", () => {
        validateInput(true); // Format valid values on blur, preserve invalid ones with error styling
    });

    el.addEventListener("input", (event) => {
        validateInput(false); // Just validate, don't correct
    }, {capture: true}); // Use capture phase to run before Alpine.js

    el.addEventListener("keydown", (event) => {
        // Handle arrow keys for step increments if step is defined
        const currentLimits = getCurrentLimits();
        let actualStepValue = currentLimits.stepvalue;
        
        
        if (actualStepValue != null) {
            let currentvalue = el.numvalue;
            if (event.key == "ArrowUp") {
                // prevent cursor from moving to start of input
                event.preventDefault();
                el.value = formatNumberInput(currentvalue + actualStepValue);
                el.dispatchEvent(new Event("input"));
                return;
            } else if (event.key == "ArrowDown") {
                // prevent cursor from moving to end of input
                event.preventDefault();
                el.value = formatNumberInput(currentvalue - actualStepValue);
                el.dispatchEvent(new Event("input"));
                return;
            }
        }
        // All other keys are allowed through naturally
    });

    // trigger blur event to trigger initial formatting
    requestAnimationFrame(() => {
        el.dispatchEvent(new Event("blur"));
    });
}
