WIKIDATA_QUERIES = {
    "base_identity": """
SELECT ?company ?isin
WHERE {
  VALUES ?company { __VALUES__ } 
  
  OPTIONAL { ?company wdt:P946 ?isin . }
}
""",
    "founding_year": """
    SELECT ?company ?founding_year
    WHERE {
    VALUES ?company { __VALUES__ } 

    OPTIONAL { ?company wdt:P571 ?founding_year . }
}""",

    "subsidiaries": """
    SELECT ?company ?subsidiaries ?subsidiaryLabel
    WHERE {
    VALUES ?company { __VALUES__ }
    OPTIONAL { ?company wdt:P355 ?subsidiaries . }
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
    "investments": """
        SELECT ?company ?investments ?investmentLabel
        WHERE {
        VALUES ?company { __VALUES__ }
        OPTIONAL { ?company wdt:P1830 ?investments . }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }""",
    "industries": """
    SELECT ?company ?industries ?industriesLabel
    WHERE {
    VALUES ?company { __VALUES__ }
    OPTIONAL { ?company wdt:P452 ?industries . }
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
    "products": """
        SELECT ?company ?products ?productsLabel
        WHERE {
        VALUES ?company { __VALUES__ }
        OPTIONAL { ?company wdt:P1056 ?products . }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }""",
    "operating_area": """
        SELECT ?company ?operating_area ?operating_areaLabel
        WHERE {
        VALUES ?company { __VALUES__ }
        OPTIONAL { ?company wdt:P2541 ?operating_area . }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }""",
    "location": """
        SELECT ?company ?location ?locationLabel
        WHERE {
        VALUES ?company { __VALUES__ }
        OPTIONAL { ?company wdt:P276 ?location . }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }""",
    "instance_of": """
        SELECT ?company ?instance_of ?instance_ofLabel
        WHERE {
        VALUES ?company { __VALUES__ }
        OPTIONAL { ?company wdt:P31 ?instance_of . }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }""",
    "part_of": """
        SELECT ?company ?part_of ?part_ofLabel
        WHERE {
        VALUES ?company { __VALUES__ }
        OPTIONAL { ?company wdt:P361 ?part_of . }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }""",
    "owned_by": """
        SELECT ?company ?owned_by ?owned_byLabel
        WHERE {
        VALUES ?company { __VALUES__ }
        OPTIONAL { ?company wdt:P127 ?owned_by . }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }"""
}
