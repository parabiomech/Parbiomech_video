# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# MediaPipe 데이터 파일 포함
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
mediapipe_datas = collect_data_files('mediapipe')
mediapipe_hiddenimports = collect_submodules('mediapipe')

a = Analysis(
    ['desktop_app.py'],
    pathex=[],
    binaries=[],
    datas=mediapipe_datas,
    hiddenimports=mediapipe_hiddenimports + [
        'cv2',
        'numpy',
        'pandas',
        'PyQt5',
        'pyqtgraph',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Parbiomech_Video_Analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI 앱이므로 콘솔창 숨김
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 아이콘 파일이 있으면 여기에 경로 추가
)
