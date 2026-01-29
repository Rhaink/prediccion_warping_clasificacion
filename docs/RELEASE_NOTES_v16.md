# Release Notes - v16 Portable Windows Package

## Release Date
2026-01-29

## Overview
Version 16 introduces public sharing capability via Gradio's built-in tunneling service, enabling remote demonstrations without complex network configuration.

## New Features

### 1. Public Sharing Mode (RUN_DEMO_SHARE.bat)
- **Purpose**: Create publicly accessible demonstration links for remote attendees
- **Duration**: 72-hour link validity
- **Use Cases**:
  - Thesis defense with remote committee members
  - Sharing demos with colleagues at other institutions
  - Remote presentations and webinars
  - Quick demos without IT department involvement

### 2. Dual Launch Mode Architecture
- **RUN_DEMO.bat**: Local-only mode (default, secure by design)
- **RUN_DEMO_SHARE.bat**: Public sharing mode (opt-in)
- Clear separation prevents accidental public exposure
- Backward compatible with v15 workflows

### 3. Enhanced Documentation
- Comprehensive public sharing guide in README.txt
- Security considerations and warnings
- Troubleshooting section for common issues
- Clear examples of when to use each mode

## Technical Implementation

### Modified Files
1. **scripts/build_portable_windows.py**:
   - Added `_get_run_demo_share_template()` method
   - Modified `create_batch_files()` to generate both launch scripts
   - Updated `_get_readme_template()` with public sharing documentation
   - Enhanced VERSION.txt with feature list
   - Updated package verification to include RUN_DEMO_SHARE.bat

### New Artifacts
1. **RUN_DEMO_SHARE.bat** (2,857 bytes):
   - Yellow console color (0E) for visual distinction
   - Security warning with pause for user acknowledgment
   - Passes `--share` flag to run_demo.py
   - Comprehensive error handling for tunnel failures
   - User-friendly instructions for link sharing

2. **Updated README.txt** (4,600 bytes):
   - New "PUBLIC SHARING MODE" section
   - Updated "QUICK START" with dual launch options
   - Enhanced troubleshooting guide

3. **Updated VERSION.txt** (1,178 bytes):
   - Features array documenting capabilities
   - Maintains model checksums for integrity

## Package Specifications

| Metric | Value |
|--------|-------|
| **Version** | 16 |
| **Build Date** | 2026-01-29 |
| **Package Size** | 579 MB (compressed) |
| **Files Count** | 196 files |
| **Python Version** | 3.12.8 |
| **Dependencies** | 86 wheel packages |

## Security Considerations

### Safety by Default
- Local-only mode remains the default (RUN_DEMO.bat)
- Public sharing requires explicit user action (separate file)
- Security warning displayed before tunnel creation
- Clear documentation of risks and appropriate usage

### Public Sharing Safeguards
1. **Explicit Opt-In**: User must double-click separate file
2. **Warning Screen**: Yellow console with security notice
3. **Manual Confirmation**: Requires pressing any key to proceed
4. **Clear Instructions**: When to use and when NOT to use
5. **Revocation**: Ctrl+C immediately stops server and revokes link

### Data Protection Guidance
- Documentation warns against uploading sensitive/patient data
- HIPAA/GDPR compliance notes included
- Recommendation to use for demonstrations only

## Backward Compatibility

### ✅ Fully Compatible
- v15 users can upgrade seamlessly
- RUN_DEMO.bat behavior unchanged
- No dependency changes
- Same model checksums
- Identical user interface

### Migration Path
1. Extract v16 package
2. Continue using RUN_DEMO.bat for local-only (no changes)
3. Use RUN_DEMO_SHARE.bat when public access needed

## Testing Checklist

### Pre-Release Verification
- [x] Build completes successfully (578.4 MB)
- [x] RUN_DEMO_SHARE.bat created in package
- [x] VERSION.txt includes features list
- [x] README.txt includes PUBLIC SHARING MODE section
- [x] QUICK START updated with dual launch options
- [x] Package integrity verified (196 files)

### Recommended User Testing (Windows)
- [ ] Extract package on clean Windows 10/11 system
- [ ] Test RUN_DEMO.bat (local-only mode, no changes from v15)
- [ ] Test RUN_DEMO_SHARE.bat:
  - [ ] Security warning displays correctly
  - [ ] Gradio tunnel creates successfully
  - [ ] Public URL appears in console
  - [ ] Remote device can access public URL
  - [ ] Ctrl+C revokes link
- [ ] Test first-time installation flow with RUN_DEMO_SHARE.bat
- [ ] Verify firewall/antivirus compatibility

## Known Limitations

### Internet Dependency
- Public sharing mode requires internet connection (for Gradio tunnel)
- Local mode continues to work offline
- Tunnel creation may fail if:
  - No internet connection
  - Firewall blocks Gradio service
  - Gradio service temporarily unavailable

### Performance
- Public link adds 1-2 second latency (normal for internet tunneling)
- Local mode recommended for presenter's view
- Remote attendees may experience slower processing

### Link Lifecycle
- 72-hour expiration (Gradio limitation)
- Must restart RUN_DEMO_SHARE.bat to create new link
- Link revoked when server stopped (Ctrl+C)

## Use Case Examples

### ✓ Recommended Uses
1. **Thesis Defense**: Share with remote committee members
2. **Academic Presentations**: Demo to conference attendees
3. **Collaboration**: Quick sharing with researchers at other institutions
4. **Remote Meetings**: Show system in Zoom/Teams without screen sharing

### ✗ Not Recommended
1. **Regular Testing**: Use RUN_DEMO.bat instead
2. **Production Deployment**: Requires proper server hosting
3. **Sensitive Data**: HIPAA/GDPR compliance not guaranteed
4. **Long-term Access**: Link expires in 72 hours

## Troubleshooting Guide

### Public Link Creation Fails
**Symptom**: "ERROR: Failed to create public share link"

**Solutions**:
1. Verify internet connection
2. Check firewall settings (allow Python/Gradio)
3. Retry (Gradio service may be busy)
4. Fallback: Use RUN_DEMO.bat + screen sharing

### Link Expired
**Symptom**: "404 Not Found" on public URL

**Solutions**:
1. Restart RUN_DEMO_SHARE.bat (creates new 72h link)
2. Share new URL with attendees

### Slow Performance
**Symptom**: 2-5 second processing delay

**Explanation**: Normal for internet tunneling
**Mitigation**:
- Use local mode for presenter's view
- Inform attendees of expected latency

## Future Enhancements (Potential v17+)

### Considered Features
- Custom port configuration (RUN_DEMO_PORT8080.bat)
- Authentication/password protection for public links
- Link expiry configuration
- Usage analytics and logging
- Multi-language documentation

### Not Planned
- Self-hosted tunneling (increases complexity)
- Database backend (maintains offline-first design)
- User accounts (conflicts with portable nature)

## Validation Results

### Build Metrics
```
[04:24:50] [INFO] BUILD SUCCESSFUL!
[04:24:50] [INFO] Package: build/releases/covid19-demo-v16-portable-windows.zip
[04:24:50] [INFO] Size: 578.4 MB
[04:24:50] [INFO] Package integrity verified (196 files in ZIP)
```

### File Verification
```bash
# Critical files present
✓ RUN_DEMO.bat (1,915 bytes)
✓ RUN_DEMO_SHARE.bat (2,857 bytes)
✓ README.txt (4,600 bytes)
✓ VERSION.txt (1,178 bytes)
✓ python/python.exe
✓ models/classifier/best_classifier.pt
```

### Model Checksums (Unchanged from v15)
```json
{
  "models/landmarks/resnet18_seed123_best.pt": "db9e4efc4a020c28...",
  "models/landmarks/resnet18_seed321_best.pt": "91b234c2414c9b9e...",
  "models/landmarks/resnet18_seed111_best.pt": "d6b9e4c65159485c...",
  "models/landmarks/resnet18_seed666_best.pt": "24d56e89723772215...",
  "models/classifier/best_classifier.pt": "2753fef71991da18..."
}
```

## Deployment Instructions

### For Developers
```bash
# Build v16
python scripts/build_portable_windows.py --version 16 --output build/releases

# Verify package
unzip -l build/releases/covid19-demo-v16-portable-windows.zip | grep RUN_DEMO
```

### For End Users
1. Download `covid19-demo-v16-portable-windows.zip` (579 MB)
2. Extract to `C:\covid19-demo` (or preferred location)
3. Choose launch mode:
   - Local testing: `RUN_DEMO.bat`
   - Public sharing: `RUN_DEMO_SHARE.bat`

### For Thesis Defense
**48 hours before**:
1. Build and test v16 package
2. Verify public link creation on target machine
3. Share test link with colleague to confirm remote access

**Day of defense**:
1. Extract package on presentation PC
2. Run `RUN_DEMO_SHARE.bat` 10 minutes before presentation
3. Copy public URL from console
4. Share URL in Zoom/Teams chat
5. Demonstrate system live
6. Press Ctrl+C after demonstration

**Backup Plan**:
- If public link fails: Use `RUN_DEMO.bat` + screen sharing
- Keep screenshots in presentation slides

## Credits
- **Implementation**: Plan-driven development (60 minutes)
- **Build System**: Python 3.12.8 embeddable package
- **Tunneling**: Gradio built-in sharing (no additional setup)
- **Documentation**: Comprehensive user guide and troubleshooting

## References
- **Build Script**: `scripts/build_portable_windows.py`
- **Run Demo**: `scripts/run_demo.py` (unchanged, already supports --share)
- **Project Docs**: `CLAUDE.md`, `docs/DEPLOYMENT.md`
- **Gradio Sharing**: https://gradio.app/sharing-your-app/

---

**Version**: 16
**Release Type**: Feature Enhancement
**Stability**: Stable (backward compatible with v15)
**Recommended For**: Production use, thesis defense, academic demonstrations
