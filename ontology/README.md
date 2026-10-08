# Ontologies used in the Gallstone mapping

The mapping experiment consulted the following OBO Foundry ontologies.

| Ontology | Role in the mapping | Canonical download |
|---|---|---|
| Clinical Measurement Ontology (CMO) | Primary clinical measurement ontology | http://purl.obolibrary.org/obo/cmo.obo |
| Mondo Disease Ontology (MONDO) | Disease / diagnosis terms | http://purl.obolibrary.org/obo/mondo.obo |
| Ontology of Biological Attributes (OBA) | Biological attributes / traits | http://purl.obolibrary.org/obo/oba.obo |
| Phenotype And Trait Ontology (PATO) | Phenotypic qualities | http://purl.obolibrary.org/obo/pato.obo |
| Vertebrate Trait Ontology (VT) | Vertebrate traits | http://purl.obolibrary.org/obo/vt.owl |

OBO Foundry pages:

- CMO: https://obofoundry.org/ontology/cmo.html
- MONDO: https://obofoundry.org/ontology/mondo.html
- OBA: https://obofoundry.org/ontology/oba.html
- PATO: https://obofoundry.org/ontology/pato.html
- VT: https://obofoundry.org/ontology/vt

For a paper release, **freeze the ontology versions** used for the final run rather
than depending only on moving PURLs. Run:

```bash
python ontology/download_ontologies.py
```

The script saves the files and writes SHA-256 checksums. Commit the frozen files
if repository size permits; otherwise deposit them in Zenodo/OSF and record the DOI
and checksums here.
