from pathlib import Path


def site_sort_key(site_id):
    """
    Sort numeric site IDs numerically and letter/string site IDs alphabetically.

    Examples:
        1, 2, 10 -> 1, 2, 10
        "A", "B" -> "A", "B"
    """
    text = str(site_id)

    if text.isdigit():
        return (0, int(text))

    return (1, text)


def normalize_site_path_map(path_map, *, name):
    """
    Normalize a site-id -> path mapping.

    Supports site IDs such as:
        1, "1", "A", "B"

    Numeric strings are converted to integers.
    Non-numeric IDs are kept as strings.
    """
    if path_map is None:
        return None

    normalized = {}

    for site_id, path in path_map.items():
        text = str(site_id)

        if text.isdigit():
            key = int(text)
        else:
            key = site_id

        normalized[key] = Path(path)

    if not normalized:
        raise ValueError(f"{name} was provided but is empty.")

    return normalized


def sorted_site_ids_from_map(path_map):
    return sorted(path_map.keys(), key=site_sort_key)


def get_path_for_site(path_map, site_id, *, name):
    if path_map is None:
        raise ValueError(f"{name} was not provided.")

    candidates = [site_id, str(site_id)]

    text = str(site_id)
    if text.isdigit():
        candidates.append(int(text))

    for key in candidates:
        if key in path_map:
            return Path(path_map[key])

    raise KeyError(f"No path configured for site {site_id} in {name}.")


def site_dir_from_root_or_map(results_root_dir, site_id, path_map=None):
    """
    Return an FL site result folder.

    """
    if path_map is not None:
        return get_path_for_site(path_map, site_id, name="fl_site_results_dirs")

    return Path(results_root_dir) / f"fl_site_{site_id}"