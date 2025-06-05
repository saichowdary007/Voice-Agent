#!/usr/bin/env node

const WebSocket = require('ws');
const fs = require('fs');
const path = require('path');

const WS_URL = process.env.WS_URL || 'ws://localhost:8003/ws';
const TIMEOUT_MS = 30000; // 30 seconds timeout for connection attempts

console.log(`Connecting to ${WS_URL}...`);

// Create WebSocket connection
const ws = new WebSocket(WS_URL);
let connected = false;

// Store test results
const results = {
    connection: null,
    ping: null,
    oneByte: null,
    oneKB: null,
    tenKB: null,
    jsonEos: null,
    stringEos: null,
};

// Connect timeout
const connectTimeout = setTimeout(() => {
    if (!connected) {
        console.error('Connection timeout after 30 seconds');
        process.exit(1);
    }
}, TIMEOUT_MS);

// Handle connection events
ws.on('open', () => {
    console.log('✅ Connection established');
    connected = true;
    clearTimeout(connectTimeout);
    results.connection = 'success';
    
    runTests();
});

ws.on('message', (data) => {
    let parsedData;
    
    try {
        if (data instanceof Buffer) {
            console.log(`Received binary message: ${data.length} bytes`);
            return;
        }
        
        parsedData = JSON.parse(data.toString());
        console.log('📨 Received:', parsedData);
        
        // Check for ping response
        if (parsedData.type === 'pong') {
            console.log('✅ Ping test: success');
            results.ping = 'success';
        }
    } catch (e) {
        console.log(`Received raw text: ${data.toString()}`);
    }
});

ws.on('error', (error) => {
    console.error('❌ WebSocket error:', error);
    results.connection = 'error: ' + error.message;
});

ws.on('close', (code, reason) => {
    console.log(`❌ Connection closed. Code: ${code}, Reason: ${reason || 'None'}`);
    
    // Log test summary
    console.log('\n=== TEST SUMMARY ===');
    for (const [test, result] of Object.entries(results)) {
        console.log(`${test}: ${result || 'not run'}`);
    }
    
    if (code !== 1000) {
        process.exit(1);
    }
    
    process.exit(0);
});

// Test sending different payloads
async function runTests() {
    // Wait between tests
    const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));
    
    try {
        // 1. Send ping
        console.log('\n1. Testing ping...');
        ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
        await wait(1000);
        
        // 2. Send 1 byte binary
        console.log('\n2. Testing 1-byte binary...');
        ws.send(new Uint8Array([42]));
        results.oneByte = 'sent';
        await wait(1000);
        
        // 3. Send 1KB binary
        console.log('\n3. Testing 1KB binary...');
        const oneKB = new Uint8Array(1024);
        for (let i = 0; i < oneKB.length; i++) oneKB[i] = i % 256;
        ws.send(oneKB);
        results.oneKB = 'sent';
        await wait(1000);
        
        // 4. Send 10KB binary
        console.log('\n4. Testing 10KB binary...');
        const tenKB = new Uint8Array(10 * 1024);
        for (let i = 0; i < tenKB.length; i++) tenKB[i] = i % 256;
        ws.send(tenKB);
        results.tenKB = 'sent';
        await wait(1000);
        
        // 5. Send JSON EOS
        console.log('\n5. Testing JSON EOS...');
        ws.send(JSON.stringify({ type: 'eos' }));
        results.jsonEos = 'sent';
        await wait(1000);
        
        // 6. Send string EOS
        console.log('\n6. Testing string EOS...');
        ws.send('EOS');
        results.stringEos = 'sent';
        await wait(1000);
        
        // 7. Close connection
        console.log('\n7. Closing connection...');
        ws.close(1000, 'Tests completed successfully');
    } catch (error) {
        console.error('Error during tests:', error);
        ws.close(1000, 'Tests failed');
    }
} 