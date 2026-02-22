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



}

   