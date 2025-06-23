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

    /** @type {number|null} */
    let minvalue = parseFloat(el.min);
    if (isNaN(minvalue)) minvalue = null;
    /** @type {number|null} */
    let maxvalue = parseFloat(el.max);
    if (isNaN(maxvalue)) maxvalue = null;
    /** @type {number|null} */
    let stepvalue = parseFloat(el.step);
    if (isNaN(stepvalue)) stepvalue = null;

    const numdigits = getnumberofdecimaldigits(stepvalue);
    /**
     *
     * @param {number} newval
     * @returns {string}
     */
    function formatNumberInput(newval) {
        if (!isFinite(newval)) {
            return (el.value || minvalue || maxvalue || 0) + "";
        }
        if (maxvalue != null && newval > maxvalue) {
            return maxvalue.toFixed(numdigits);
        }
        if (minvalue != null && newval < minvalue) {
            return minvalue.toFixed(numdigits);
        }
        if (getnumberofdecimaldigits(newval) > numdigits) {
            return newval.toFixed(numdigits);
        }

        return parseFloat(newval + "").toFixed(numdigits);
    }

    const maxnumchars = Math.max(
        formatNumberInput(minvalue ?? 0).length,
        formatNumberInput(maxvalue ?? 0).length,
        formatNumberInput(el.numvalue).length,
    );
    if (maxnumchars != null && !isNaN(maxnumchars)) {
        el.setAttribute("size", `${maxnumchars}`);
    }

    let placeholdervalue = parseFloat(el.getAttribute("placeholder") ?? "");
    if (!isNaN(placeholdervalue)) {
        el.setAttribute("placeholder", formatNumberInput(placeholdervalue));
    }

    el.addEventListener("blur", () => {
        /** @type {number} */
        const currentValue = el.numvalue;

        // if input is valid, write it back.
        // otherwise, (if value is nan, e.g. when the user empties the input)
        // set the value to the placeholder (default) value.
        if (!isNaN(currentValue)) {
            el.value = formatNumberInput(currentValue);
        } else {
            if (isNaN(placeholdervalue))
                console.warn("empty input without placeholder");
            el.value = formatNumberInput(placeholdervalue);
        }
    });

    el.addEventListener("keydown", (event) => {
        // check for valid number of digits
        let keyisnumericinput = "0123456789e+-.".indexOf(event.key) != -1;
        let keyismetainput =
            event.ctrlKey ||
            event.metaKey ||
            [
                "ArrowLeft",
                "ArrowRight",
                "ArrowUp",
                "ArrowDown",
                "Tab",
                "Backspace",
                "Delete",
                "Enter",
                "End",
                "Home",
            ].indexOf(event.key) != -1;

        if (!(keyisnumericinput || keyismetainput)) {
            // ignore whatever would happen here
            event.preventDefault();
            return;
        }

        if (keyismetainput) {
            // do not prevent input here (just ignore and propagate)

            if (stepvalue != null) {
                let currentvalue = el.numvalue;
                if (event.key == "ArrowUp") {
                    // prevent cursor from moving to start of input
                    event.preventDefault();
                    el.value = formatNumberInput(currentvalue + stepvalue);

                    // flush change
                    el.dispatchEvent(new Event("input"));
                    return;
                } else if (event.key == "ArrowDown") {
                    // prevent cursor from moving to end of input
                    event.preventDefault();
                    el.value = formatNumberInput(currentvalue - stepvalue);

                    // flush change
                    el.dispatchEvent(new Event("input"));
                    return;
                }
            }
            return;
        }

        // we handle input ourselves, no need for default
        event.preventDefault();

        let eventTarget = event.currentTarget;
        if (!(eventTarget instanceof HTMLInputElement)) throw new Error("");
        let selectionstart = eventTarget.selectionStart ?? 0;
        let selectionend = eventTarget.selectionEnd ?? eventTarget.value.length;

        let modifiedinput =
            el.value.toString().substring(0, selectionstart) +
            event.key +
            el.value.toString().substring(selectionend);

        // ignore input if change makes number invalid
        if (
            parseFloat(modifiedinput) !=
            parseFloat(formatNumberInput(parseFloat(modifiedinput)))
        ) {
            return;
        }

        el.value = modifiedinput; //formatNumberInput(parseFloat(modifiedinput));
        eventTarget.setSelectionRange(selectionstart + 1, selectionstart + 1);

        // trigger input event to e.g. flush x-model on same element
        el.dispatchEvent(new Event("input"));
    });

    // trigger blur event to trigger initial formatting
    requestAnimationFrame(() => {
        el.dispatchEvent(new Event("blur"));
    });
}
