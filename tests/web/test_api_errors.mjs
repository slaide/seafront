import test from 'node:test';
import assert from 'node:assert/strict';
import { APIClient } from '../../seafront/web-static/src/api-client.js';

/** @returns {{client: APIClient, shown: [string, string][]}} */
function client() {
    /** @type {[string, string][]} */
    const shown = [];
    const api = new APIClient('http://scope', { onError: (title, detail) => { shown.push([title, detail]); } });
    return { client: api, shown };
}

test('a plain-text 500 shows its text, not a JSON parse error', async () => {
    // what a server without the JSON error handler sends for an unhandled OSError
    const { client: api, shown } = client();
    const response = new Response('Internal Server Error', { status: 500, statusText: 'Internal Server Error' });
    await assert.rejects(api.handleResponse(response, 'Start acquisition'), (error) => {
        assert.ok(error instanceof Error);
        assert.doesNotMatch(error.message, /JSON/);
        assert.match(error.message, /Internal Server Error/);
        return true;
    });
    assert.deepEqual(shown, [['Start acquisition Error', 'Internal Server Error']]);
});

test('a JSON 500 shows its detail', async () => {
    const { client: api, shown } = client();
    const detail = 'cannot start the acquisition: the image output directory /mnt/share cannot be reached';
    const response = new Response(JSON.stringify({ detail }), { status: 500, statusText: 'Internal Server Error' });
    await assert.rejects(api.handleResponse(response, 'Start acquisition'), /cannot be reached/);
    assert.deepEqual(shown, [['Start acquisition Error', detail]]);
});

test('an empty 500 falls back to the status text', async () => {
    const { client: api, shown } = client();
    const response = new Response('', { status: 500, statusText: 'Internal Server Error' });
    await assert.rejects(api.handleResponse(response, 'Start acquisition'));
    assert.deepEqual(shown, [['Start acquisition Error', 'Internal Server Error']]);
});
