"""Small local SVG icon set used by the desktop UI."""

from PyQt5.QtCore import QByteArray, QSize, Qt
from PyQt5.QtGui import QIcon, QPainter, QPixmap

try:
    from PyQt5.QtSvg import QSvgRenderer
except ImportError:  # pragma: no cover - PyQt5 desktop installs normally include QtSvg
    QSvgRenderer = None


_PATHS = {
    "camera": '<rect x="3" y="7" width="18" height="13" rx="2"/><path d="M8 7l1.5-3h5L16 7"/><circle cx="12" cy="13.5" r="3.5"/>',
    "video": '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="m16 10 5-3v10l-5-3z"/>',
    "image": '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8" cy="9" r="1.5"/><path d="m4 17 5-5 4 4 2-2 6 6"/>',
    "stop": '<rect x="5" y="5" width="14" height="14" rx="2"/>',
    "save": '<path d="M5 3h12l3 3v15H4V3z"/><path d="M8 3v6h8V3M8 21v-7h8v7"/>',
    "settings": '<path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6 7 7M17 17l1.4 1.4M18.4 5.6 17 7M7 17l-1.4 1.4"/><circle cx="12" cy="12" r="4"/>',
    "volume": '<path d="M4 10v4h4l5 4V6l-5 4H4z"/><path d="M16 9a5 5 0 0 1 0 6M18.5 6.5a9 9 0 0 1 0 11"/>',
    "volume-x": '<path d="M4 10v4h4l5 4V6l-5 4H4z"/><path d="m17 9 4 4m0-4-4 4"/>',
    "ruler": '<path d="m4 16 12-12 4 4L8 20H4z"/><path d="m12 8 4 4M9 11l3 3M6 14l3 3"/>',
    "scan": '<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><path d="M7 12h10M12 7v10"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 1v5M15 1v5M9 18v5M15 18v5M1 9h5M1 15h5M18 9h5M18 15h5"/><path d="M10 10h4v4h-4z"/>',
    "alert": '<path d="m12 3 9 17H3L12 3z"/><path d="M12 9v4M12 17h.01"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16 9"/>',
}


def icon(name, color="#CBD5E1", size=18):
    if QSvgRenderer is None or name not in _PATHS:
        return QIcon()
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" '
        'stroke-linecap="round" stroke-linejoin="round">'
        f"{_PATHS[name]}</svg>"
    )
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)
