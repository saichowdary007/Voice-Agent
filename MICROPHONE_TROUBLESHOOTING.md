# Microphone Access Troubleshooting Guide

## Common Issues & Solutions

### 1. **Browser Permission Denied**
**Symptoms:** Console shows "NotAllowedError" or permission denied
**Solutions:**
- Click the 🔒 icon in browser address bar
- Allow microphone access for localhost
- Refresh the page after allowing permissions

### 2. **No Microphone Found**
**Symptoms:** Console shows "NotFoundError" 
**Solutions:**
- Ensure microphone is connected and working
- Check System Preferences → Sound → Input
- Try a different microphone

### 3. **HTTPS Requirement** 
**Symptoms:** getUserMedia fails on non-localhost domains
**Solutions:**
- Use localhost for development (already doing this ✅)
- For production, ensure HTTPS is enabled

### 4. **Browser Compatibility**
**Symptoms:** "getUserMedia not supported"
**Solutions:**
- Use Chrome, Firefox, Safari, or Edge (modern versions)
- Avoid older browsers or incognito mode restrictions

### 5. **Audio Context Issues**
**Symptoms:** WebSocket connects but no audio chunks sent
**Solutions:**
- Check if AudioContext is suspended (browser autoplay policy)
- User interaction may be required before audio works

## Testing Steps

1. **Open Browser Console** (F12)
2. **Navigate to** `http://localhost:3000`
3. **Click "Start New Session"**
4. **Look for these console messages:**
   - 🎤 Starting microphone access request...
   - 🔍 Checking available audio devices...
   - ✅ Microphone access granted!
   - 🔊 useEffect: Connection active, attempting to initialize audio
   - 🎯 Audio initialization succeeded

## Console Command Tests

Run these in the browser console to test:

```javascript
// Test 1: Check getUserMedia support
console.log('getUserMedia supported:', !!navigator.mediaDevices?.getUserMedia);

// Test 2: List audio devices
navigator.mediaDevices.enumerateDevices()
  .then(devices => console.log('Audio inputs:', devices.filter(d => d.kind === 'audioinput')));

// Test 3: Test basic microphone access
navigator.mediaDevices.getUserMedia({audio: true})
  .then(stream => {
    console.log('✅ Basic mic access OK:', stream);
    stream.getTracks().forEach(track => track.stop());
  })
  .catch(err => console.error('❌ Basic mic access failed:', err));
```

## Backend Logs to Check

Look for these in the backend terminal:
- `WebSocket connection accepted`
- `Watchdog timer started`
- `Received binary message: X bytes` (when audio is sent)

If you see 30s timeout messages but no binary audio data, the issue is frontend microphone access. 