try:
    from openff.utilities import provenance as _openff_provenance  # type: ignore

    _original_get_ambertools_version = _openff_provenance.get_ambertools_version

    def _safe_get_ambertools_version():
        try:
            return _original_get_ambertools_version()
        except TypeError:
            return None

    _openff_provenance.get_ambertools_version = _safe_get_ambertools_version
except Exception:
    pass
