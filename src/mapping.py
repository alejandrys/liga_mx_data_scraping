import unicodedata

def normalize_base(s):
    if not isinstance(s, str):
        return s

    s = s.lower().strip()

    # quitar acentos
    s = ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

    return s


TEAM_MAP = {
    # América
    "america": "america",
    "club america": "america",

    # Guadalajara
    "guadalajara": "guadalajara",
    "guadalajara chivas": "guadalajara",
    "chivas": "guadalajara",

    # Juárez
    "fc juarez": "juarez",
    "juarez": "juarez",

    # San Luis
    "atletico san luis": "san luis",

    # Mazatlán
    "mazatlan": "mazatlan",
    "mazatlan fc": "mazatlan",

    # Querétaro
    "club queretaro": "queretaro",
    "queretaro": "queretaro",

    # Tijuana
    "club tijuana": "tijuana",
    "tijuana": "tijuana",

    # Pumas
    "u.n.a.m. - pumas": "pumas",
    "pumas": "pumas",

    # Tigres
    "tigres uanl": "tigres",
    "tigres": "tigres",

    # León
    "leon": "leon",

    # Atlas
    "atlas": "atlas",

    # Monterrey
    "monterrey": "monterrey",

    # Cruz Azul
    "cruz azul": "cruz azul",

    # Toluca
    "toluca": "toluca",

    # Pachuca
    "pachuca": "pachuca",

    # Puebla
    "puebla": "puebla",

    # Santos
    "santos laguna": "santos laguna",

    # Necaxa
    "necaxa": "necaxa"
}


def map_team(s):
    s = normalize_base(s)
    return TEAM_MAP.get(s, s)