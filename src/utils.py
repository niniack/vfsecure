# Utility script for cross-file tools

def _b(s, encoding='ascii', errors='replace'):
	if isinstance(s, str):
		return bytes(s, encoding, errors)
	else:
		return s
