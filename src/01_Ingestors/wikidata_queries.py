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
    SELECT ?company ?subsidiaries 
    WHERE {
    VALUES ?company { __VALUES__ }
    OPTIONAL { ?company wdt:P355 ?subsidiaries . }
}""",
    "investments": """
        SELECT ?company ?investments 
        WHERE {
        VALUES ?company { __VALUES__ }
        OPTIONAL { ?company wdt:P1830 ?investments . }
    }""",
    "industries": """
    SELECT ?company ?industries 
    WHERE {
    VALUES ?company { __VALUES__ }
    OPTIONAL { ?company wdt:P452 ?industries . }
}""",
    "products": """
        SELECT ?company ?products 
        WHERE {
        VALUES ?company { __VALUES__ }
        OPTIONAL { ?company wdt:P1056 ?products . }
    }""",
    "operating_area": """
        SELECT ?company ?operating_area 
        WHERE {
        VALUES ?company { __VALUES__ }
        OPTIONAL { ?company wdt:P2541 ?operating_area . }
    }""",
    "location": """
        SELECT ?company ?location 
        WHERE {
        VALUES ?company { __VALUES__ }
        OPTIONAL { ?company wdt:P276 ?location . }
    }""",
    "instance_of": """
        SELECT ?company ?instance_of 
        WHERE {
        VALUES ?company { __VALUES__ }
        OPTIONAL { ?company wdt:P31 ?instance_of . }
    }""",
    "part_of": """
        SELECT ?company ?part_of 
        WHERE {
        VALUES ?company { __VALUES__ }
        OPTIONAL { ?company wdt:P361 ?part_of . }
    }""",
    "owned_by": """
        SELECT ?company ?owned_by 
        WHERE {
        VALUES ?company { __VALUES__ }
        OPTIONAL { ?company wdt:P127 ?owned_by . }
    }"""
}
