# Downloadable mouse developmental and spatial datasets

Research date: **29 October 2026**.

This catalog contains **53 dataset/atlas entries**, followed by additional databases and discovery routes. It prioritizes mouse gastrulation and early organogenesis, then later development, with a smaller adult reference section. It is an extensive curated starting point, not an exhaustive inventory of every public mouse experiment. Integrated atlases and their constituent studies are listed separately where each offers a useful download, but they are **not independent datasets**.

The anchor study is Xie et al., **“Digital reconstruction of full embryos during early mouse organogenesis”**, *Cell* (2025), [DOI: 10.1016/j.cell.2025.05.035](https://doi.org/10.1016/j.cell.2025.05.035). Its sampled stages are **E7.5, E7.75, and E8.0**, with two embryos per stage. Most entries below concern the same biological processes, nearby stages, or subsequent development of the same lineages. “Related” means biological or methodological relevance; it does not imply that every study was cited or used by Xie et al. Later studies cannot have been inputs to the 2025 paper.

## Reading the catalog

- **E** denotes embryonic day; **P** denotes postnatal day. A stage range does not imply every intermediate stage was sampled. Exact stages are listed where established.
- **Measured spatial** means expression was measured with tissue position retained. A reconstructed 3D embryo usually combines measured sections with computational registration.
- **Inferred spatial** means dissociated cells or unmeasured genes were assigned to space computationally. This is different from directly measuring their positions or expression.
- **Single-cell / single-nucleus** expression generally has no original cell coordinates. UMAP coordinates are not tissue coordinates.
- **Processed download** means the source advertises matrices or analysis objects. **Raw download** means sequencing reads requiring preprocessing. Links were researched from publications, official project pages, and repository records; the full files were not downloaded or integrity-tested for this catalog.
- GEO archives called `RAW.tar` often contain **processed supplementary files**, not raw sequencing reads. Inspect their contents; raw reads are usually in SRA.

## Best starting points for this project

| Priority | Resource | Stage match | Why start here? |
| --- | --- | --- | --- |
| 1 | **Xie / SEU-3D**, entry 1 | E7.5, E7.75, E8.0 | The exact README dataset; spatial `.h5ad` files and reconstructed embryos. |
| 2 | **Pijuan-Sala gastrulation atlas**, entry 20 | All three target stages | Dense single-cell reference for annotating cells and selecting genes. |
| 3 | **Extended Mouse Atlas**, entry 21 | All three target stages, through E9.5 | Convenient `.h5ad` download; extends the preceding atlas. |
| 4 | **TOME**, entry 19 | Separate E7.5, E7.75, E8.0 downloads | Convenient stage-specific references and developmental trajectories. |
| 5 | **mdCLA**, entry 39 | Separate E7.5, E7.75, E8.0 downloads | Additional developmental reference with stage-specific `.h5ad` counts. |
| 6 | **Harland spatial atlas**, entry 3 | Measured E7.5; nearby E7.25 and E8.5 | Independent imaging-based spatial context; separate observed and imputed expression. |
| 7 | **Mittnenzweig atlas**, entry 23 | E6.5–E8.25 collection window | Individual embryo identities and fine temporal ordering. |
| 8 | **Tyser heart atlas**, entry 30 | Approximately E7.5–E8.0 | Especially relevant to cardiac primordium formation in the anchor paper. |
| 9 | **MOSTA**, entry 2 | E9.5–E16.5 | Extensive measured spatial follow-up through organogenesis. |
| 10 | **MELA**, entry 26 | E7.5 and E8.0, then half-day intervals | Expression plus experimental lineage information. |
| 11 | **NextCell**, entry 27 | E13.5 endpoint | Experimental lineage history; not expression snapshots at the target stages. |

## A. Developmental spatial transcriptomics

### 1. Digital embryo / SEU-3D — Xie et al., 2025

- **Stages and tissue:** E7.5, E7.75, E8.0; six whole embryos, two per stage.
- **Modality:** Stereo-seq serial sections with single-cell assignment and 3D reconstruction. The published atlas reports 104,343 cells.
- **Download:** [GEO GSE278603](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278603); [direct `GSE278603_RAW.tar`](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278603/suppl/GSE278603_RAW.tar), approximately 803 MiB, containing six `.h5ad` files. [SEU-3D code](https://github.com/RainyBlue-w/SEU-3D).
- **Relevance:** Exact anchor dataset. The adjacent [load_dataset.py](load_dataset.py) downloads and loads this archive. Preserve embryo identity, section identity, original expression layers, and available coordinate fields when combining files.

### 2. MOSTA — Mouse Organogenesis Spatiotemporal Transcriptomic Atlas

- **Stages and tissue:** E9.5, E10.5, E11.5, E12.5, E13.5, E14.5, E15.5, E16.5; whole-embryo sagittal sections.
- **Modality:** Stereo-seq; 53 sections in the original Chen et al. 2022 atlas. Spatial units depend on the downloaded binning/segmentation product.
- **Download:** [MOSTA portal](https://db.cngb.org/stomics/mosta/), [processed downloads](https://db.cngb.org/stomics/mosta/download/), including section-specific `.h5ad` files. [Study DOI](https://doi.org/10.1016/j.cell.2022.04.003); raw project **CNP0001543**.
- **Relevance:** Strong continuation of whole-embryo spatial analysis into later organogenesis. It does **not** sample E7.5–E8.0. The download page also hosts later resources, so distinguish the original 53 sections from other studies such as entry 9.

### 3. Spatiotemporal atlas of mouse gastrulation and early organogenesis — Harland et al., 2025

- **Stages:** Measured seqFISH at **E7.25 and E7.5**, integrated with published **E8.5** seqFISH and E6.5–E9.5 scRNA-seq.
- **Modality:** Targeted imaging-based spatial transcriptomics, integrated cell annotations, and genome-wide expression imputation.
- **Download:** [Project and download instructions](https://github.com/ltgharland/Spatiotemporal-Atlas-of-Mouse-Gastrulation), [atlas/download portal](https://shiny.maths.usyd.edu.au/SpatiotemporalMouseAtlas/), [portal source with download links](https://github.com/SydneyBioX/SpatiotemporalMouseAtlas). Products include observed seqFISH data, imputed logcounts, and joint annotations.
- **Relevance:** Particularly close spatial comparator. **Staging caveat:** some project pages retain E6.5 collection labels; the [published paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7618461/) identifies those embryos morphologically as E7.25. It shares the older E8.5 data with entry 4 and single-cell data with entry 21.

### 4. Spatial Mouse Atlas / seqFISH — Lohoff et al., 2022

- **Stage and tissue:** E8.5, approximately 8–12 somites; embryo sections.
- **Modality:** seqFISH with a designed 387-gene panel, integrated with single-cell reference atlases. Genome-wide imputed expression is a separate product from the measured panel.
- **Download:** [Author data directory](https://content.cruk.cam.ac.uk/jmlab/SpatialMouseAtlas2020/); alternatively the [MouseGastrulationData package](https://bioconductor.org/packages/release/data/experiment/html/MouseGastrulationData.html) provides a curated access route. [Paper](https://www.nature.com/articles/s41587-021-01006-2).
- **Relevance:** Close temporal neighbor and useful panel-selection benchmark. Not an independent validation set if training already uses the same seqFISH data through entry 3.

### 5. eGastrulation / Geo-seq — Peng et al., 2019

- **Stages and tissue:** E5.5–E7.5; spatially dissected early embryos and germ layers.
- **Modality:** Geo-seq of small regional cell populations, approximately 20–40 cells per sample; spatial positions are anatomical regions rather than segmented single cells.
- **Download:** [GSE120963](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120963), with compressed expression tables and sample metadata; [paper and data availability](https://www.nature.com/articles/s41586-019-1469-8).
- **Relevance:** Directly useful for gastrulation patterning and regional mesoderm/endoderm identity. GEO notes that left/right labels in mirrored images must be interpreted appropriately.

### 6. Time-, space-, and single-cell-resolved gastrulation atlas — Wang et al., 2023

- **Stages:** E6.5, E6.75, E7.0, E7.25, E7.5.
- **Modality:** Geo-seq plus scRNA-seq and inferred localization of dissociated cells; includes left–right patterning information.
- **Download:** [GSE171588](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171588); [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10502153/).
- **Relevance:** Quarter-day sampling up to the earliest target stage. The integrated spatial reference incorporates earlier Geo-seq data from entry 5; do not count those reused measurements twice.

### 7. Early spatial epiblast / Geo-seq precursor — Peng et al., 2016

- **Stage:** E7.0 mid-gastrulation.
- **Modality:** Laser microdissection with positional annotation and low-cell-number RNA-seq; includes single-cell comparison data.
- **Download:** [GSE65924](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65924), including compressed regional expression tables and an E7.0 single-cell expression table.
- **Relevance:** Earlier regionalization of the same germ-layer programs. The historical GEO title says “Stereo-sequencing”; this is **not** the later DNA-nanoball Stereo-seq platform used in MOSTA and the anchor study.

### 8. Whole-embryo Slide-seq atlas — Sampath Kumar et al., 2023

- **Stages:** E8.5, E9.0, E9.5.
- **Modality:** Transcriptome-wide Slide-seq spatial maps, single-cell reference mapping, and 3D reconstruction for selected embryos.
- **Download:** [GSE197353](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE197353), [CELLxGENE collection](https://cellxgene.cziscience.com/collections/d74b6979-efba-47cd-990a-9d80ccf29055), and author-linked [E8.5 embryo 2 `.h5ad`](https://figshare.com/articles/dataset/E8_5_Embryo2_h5ad/21695849/1) and [E9.0 `.h5ad`](https://figshare.com/articles/dataset/E9_0_Embryo_h5ad/21695879/1).
- **Relevance:** Nearby whole-embryo spatial reference with cross-platform value. See [paper and download statement](https://pmc.ncbi.nlm.nih.gov/articles/PMC10335937/). Bead measurements are not necessarily one cell per bead.

### 9. Spateo whole-embryo 3D resources — Qiu et al., 2024

- **Stages:** E9.5 and E11.5; whole embryos and heart subsets.
- **Modality:** Stereo-seq with reconstructed 3D coordinates.
- **Download:** [Official Spateo data page](https://spateodata.aristoteleo.com/) offers `Mouse_E9.5_embryo.h5ad` (~822 MB), `Mouse_E11.5_embryo.h5ad` (~6.69 GB), and smaller heart subsets (~36 MB and ~138 MB). [Paper DOI](https://doi.org/10.1016/j.cell.2024.10.011).
- **Relevance:** Later whole-embryo geometry and cardiac development. The same files also appear on MOSTA's download page; mirrors and organ subsets are not additional independent specimens.

### 10. MOST / three-dimensional molecular architecture of organogenesis — Qu et al., 2023

- **Stage:** E13.5; serial sections from three embryos.
- **Modality:** 10x Visium with 3D reconstruction and single-cell deconvolution.
- **Download:** [GSE237308](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE237308); expression, spatial metadata, and section images. Additional archive **OEP003721** and portal are documented in the [paper](https://www.nature.com/articles/s41467-023-40155-7).
- **Relevance:** Whole-organ architecture and signaling after early organogenesis. **MOST and MOSTA are different resources.** Visium spots may contain multiple cells.

### 11. Visium spatiotemporal transcriptomes of mouse embryos

- **Stages:** E9.5, E10.5, E11.5, E12.5; sagittal sections.
- **Modality:** 10x Visium.
- **Download:** [GSE178636](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE178636); `GSE178636_RAW.tar` contains processed sample archives.
- **Relevance:** A smaller four-stage spatial series for later organogenesis and cross-platform comparison.

### 12. DBiT-seq embryonic spatial multi-omics — Liu et al., 2020

- **Stages:** E10, E11, E12, using the study's stage labels.
- **Modality:** Microfluidic spatial RNA profiling, with protein measurements in selected experiments; 10, 25, or 50 µm pixels depending on the sample.
- **Download:** [GSE137986](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137986), [author analysis repository](https://github.com/rongfan8/DBiT-seq), [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7736559/).
- **Relevance:** Independent spatial technology with whole-embryo and organ-level measurements. Match each sample to its actual pixel size and assay.

### 13. MPSTA — Mouse Placentation Spatiotemporal Transcriptomic Atlas

- **Stages:** E7.5–E14.5; uterus/placentation sections, including E7.5 material.
- **Modality:** Stereo-seq; extraembryonic and maternal tissue context.
- **Download:** [Processed `.h5ad` files, STDS0000245](https://db.cngb.org/stomics/datasets/STDS0000245/data), [raw project CNP0004966](https://db.cngb.org/data_resources/project/CNP0004966), [paper](https://www.nature.com/articles/s41421-024-00740-6).
- **Relevance:** Especially useful for the embryonic–extraembryonic boundary emphasized by the anchor paper. Includes high-fat-diet samples; select baseline specimens explicitly. Maternal uterine cells must not be treated as embryonic cells.

### 14. Developing stomach and intestine atlas — Zhao et al., 2022

- **Stages:** scRNA-seq at E9.5, E10.5, E11.5, E13.5, E15.5; spatial RNA-seq specifically at **E13.5 and E15.5**.
- **Modality:** Single-cell and spatial expression, with epithelial/mesenchymal annotation.
- **Download:** [GSE186525](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186525); separate `sc_UMIcounts`, `sc_meta`, `sp_UMIcounts`, and `sp_meta` tables plus image archives.
- **Relevance:** Downstream endoderm organogenesis and epithelial–mesenchymal signaling. Do not describe the entire E9.5–E15.5 interval as spatially sampled.

### 15. Embryonic endothelial spatial atlas — Chen et al., 2024

- **Stages:** MERFISH at **E9.5 and E15.5**; the accompanying single-cell study also profiles E12.5.
- **Modality:** Targeted MERFISH/MERSCOPE with an endothelial-focused panel, paired conceptually with scRNA-seq.
- **Download:** [Spatial dataset GSE247450](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE247450); [paper and companion data links](https://pmc.ncbi.nlm.nih.gov/articles/PMC10906678/).
- **Relevance:** Vascular and hematoendothelial derivatives of early mesoderm. Use the measured panel as the gene universe for spatial evaluation; do not confuse this accession with the paper's separate bulk RNA-seq series.

### 16. Developing embryo Visium dataset accompanying spatial ATAC work

- **Stages:** E12.5, E13.5, E15.5.
- **Modality:** Visium expression on embryo sections; companion epigenomic experiments are separate assays.
- **Download:** [GSE214989](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE214989), including `counts_embryo_visium.h5`, `metadata_embryo_visium.csv.gz`, and spatial/image files. Parent study: **GSE214991**.
- **Relevance:** Later-stage multi-tissue spatial benchmark. The `.h5` count file is not automatically an AnnData `.h5ad` file.

### 17. 10x Genomics Visium HD mouse embryo demonstration dataset

- **Stage:** E15.5, male C57BL/6, FFPE.
- **Modality:** Probe-based Visium HD spatial gene expression.
- **Download:** [Official dataset and output files](https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-mouse-embryo), including Space Ranger products and tissue imaging.
- **Relevance:** Convenient high-resolution whole-embryo section for pipeline development. A vendor demonstration specimen is not a replicated developmental time series, and HD bins should not automatically be equated with cells.

### 18. Whole mouse pup Xenium Prime 5K dataset

- **Stage:** P1, whole-pup section, FFPE.
- **Modality:** Xenium Prime 5K mouse pan-tissue panel with Snap25 isoform add-ons.
- **Download:** [Official dataset](https://www.10xgenomics.com/datasets/xenium-prime-ffpe-neonatal-mouse); expression matrices, transcript locations, segmentation, and morphology products are available in the output bundles.
- **Relevance:** Postnatal endpoint spanning many tissues and a useful targeted-panel comparison. Large imaging/transcript files; the assayed panel is not an unrestricted whole transcriptome.

## B. Whole-embryo and early-development single-cell references

These resources are primarily dissociated-cell or nuclear expression measurements. Developmental trajectories, lineage trees, and spatial projections are distinct kinds of information.

| # | Dataset / atlas | Stages and biological coverage | Download route and products | Relationship and practical notes |
| --- | --- | --- | --- | --- |
| 19 | **TOME — Trajectories of Mouse Embryogenesis**, Qiu et al. 2022 | 19 stages spanning E3.5–E13.5; explicit **E7.5, E7.75, E8.0** downloads | [Official portal/download section](https://tome.gs.washington.edu/); stage-specific Seurat objects with annotations; [paper](https://www.nature.com/articles/s41588-022-01018-x) | High-priority temporal reference. Integrates several studies, including entries 20, 22, 25, 36, and 37. Trajectories are inferred, not measured lineage pedigrees; not a measured spatial atlas. |
| 20 | **Mouse gastrulation atlas**, Pijuan-Sala et al. 2019 | **E6.5, E6.75, E7.0, E7.25, E7.5, E7.75, E8.0, E8.25, E8.5**; whole embryos with extraembryonic material | [Author repository and processed data instructions](https://github.com/MarioniLab/EmbryoTimecourse2018), [E-MTAB-6967](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-6967), [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6522369/) | One of the closest reference datasets. Counts, annotations, and embryo/sample metadata; also accessible through MouseGastrulationData. |
| 21 | **Extended Mouse Atlas**, Imaz-Rosshandler et al. 2024 | E6.5–E9.5, 13 quarter-day time points; 430,339 cells | [Official downloads](https://marionilab.github.io/ExtendedMouseAtlas/): `embryo_complete.h5ad`, `embryo_sce.rds`, sparse counts and metadata; raw accessions **E-MTAB-11763 / E-MTAB-13645** | Particularly convenient for Python. Extends entry 20 with new E8.5–E9.5 data; do not treat the overlapping early cells as independent validation. |
| 22 | **Somite-resolved E8.5 atlas / TOME bridge dataset** | Harvested E8.5; primitive-streak specimen and embryos staged at 2–12 somites | [GSE186069](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186069); count matrices and cell/gene annotations | New sci-RNA-seq3 data underlying part of TOME. Useful for very fine developmental staging just after the anchor window. |
| 23 | **Single-embryo time-resolved gastrulation model**, Mittnenzweig et al. 2021 | Collected E6.5–E8.25; 153 wild-type embryos, with additional perturbation experiments | [GSE169210](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE169210); MARS-seq expression and embryo metadata | Strong match to the target developmental interval. Retains individual embryo identity; chronological collection age and model-derived developmental time should be kept separate. |
| 24 | **OMG / prenatal mouse time-lapse atlas**, Qiu et al. 2024 | Main new series spans late gastrulation **E8–P0**, densely staged; larger integrated trajectory reconstruction includes earlier published data | [GSE228590](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE228590); run-wise `.mtx.gz` gene counts plus cell/gene annotation CSVs; [OMG portal](https://omg.gs.washington.edu/) | Massive nuclear-expression reference through birth. Filter using the actual sample metadata; it does not supply new E7.5/E7.75 expression measurements merely because its integrated trajectory extends earlier. |
| 25 | **MOCA — Mouse Organogenesis Cell Atlas**, Cao et al. 2019 | E9.5, E10.5, E11.5, E12.5, E13.5; approximately two million profiles from 61 embryos | [GSE119945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE119945), [paper/data availability](https://www.nature.com/articles/s41586-019-0969-x) | Broad later-organogenesis reference. Some TOME products reprocess/deepen these libraries; those are not independent biological replicates. |
| 26 | **MELA — Mouse Embryonic Lineage Atlas** | **E7.5, E8.0, E8.5, E9.0, E9.5, E10.0**; 16 embryos | [Official download page](https://mela.wi.mit.edu/download/), [Zenodo release](https://zenodo.org/records/23088914); per-embryo `.h5td` files with expression, metadata, and lineage trees | Especially relevant for E7.5/E8.0 and fate restriction. A 2026 resource; source describes the associated work as a preprint. H5TD requires TreeData-compatible tooling rather than assuming it is a plain `.h5ad`. |
| 27 | **NextCell / DNA Typewriter mouse lineage atlas** | Cells sampled at **E13.5**; lineage reconstruction extends back toward the zygote | [Official NextCell portal and downloads](https://next-cell.org/), [GSE341627](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE341627), [author code/metadata](https://github.com/shendurelab/dtt-mouse); Newick trees, count matrices, and cell metadata | User-requested resource. Distinguish experimentally recorded lineage information from inferred ancestral cell states. E0–E13.5 tree branch dates are **not** repeated RNA measurements at every earlier stage. |
| 28 | **Mouse gut endoderm emergence atlas**, Nowotschin et al. 2019 | E3.5, E4.5, E5.5, E6.5, E7.5, E8.75; embryonic and extraembryonic endoderm | [scRNA-seq GSE123046](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE123046), [paper](https://www.nature.com/articles/s41586-019-1127-1) | Very close biological match for endoderm origin and patterning. Parent **GSE123135** also contains bulk data; it is not an additional independent single-cell atlas. |
| 29 | **Early mesoderm diversification**, Scialdone et al. 2016 | E6.5–E7.75; epiblast and nascent mesoderm, including Flk1-positive populations | [E-MTAB-4079](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-4079), [E-MTAB-4026](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-4026); [author institute data links](https://www.stemcells.cam.ac.uk/news/scialdonepaper) | Smaller, focused reference for mesoderm and hematovascular fate. Sorted populations and Tal1 perturbations require careful sample selection. |
| 30 | **Mouse embryonic heart atlas / juxta-cardiac field**, Tyser et al. 2021 | Approximately **E7.5–E8.0**, morphologically staged early heart formation | [Author data directory](https://content.cruk.cam.ac.uk/jmlab/mouseEmbryonicHeartAtlas/), [analysis repository](https://github.com/MarioniLab/mouseHeart_2020), [E-MTAB-7403](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-7403); count matrices, metadata, and R objects | Particularly close to the anchor paper's cardiac primordium questions. Imaging supports anatomy, but the expression matrix is not a per-cell measured spatial transcriptome. |
| 31 | **Early cardiogenesis / Hand2 atlas**, de Soysa et al. 2019 | **E7.75**, E8.25, E9.25; cardiac progenitor regions | [GSE126128](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126128), [paper](https://www.nature.com/articles/s41586-019-1414-x); processed expression matrices and raw reads | Exact E7.75 match for cardiac lineages. Contains wild-type and Hand2-null embryos; genotype is essential metadata. |
| 32 | **Mesp1 cardiovascular progenitor atlas**, Lescroart et al. 2018 | E6.75 and E7.25 | [GSE100471](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100471); compressed HTSeq count table | Captures upstream cardiovascular lineage segregation. Mesp1-enriched cells, including mutants, are not an unbiased whole-embryo sample. |
| 33 | **Decoding gene regulation in the mouse embryo**, Argelaguet and colleagues | E7.5–E8.75; wild-type and Brachyury knockout material | [GSE205117](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205117); paired 10x Multiome RNA/accessibility, metadata and regulatory tracks | Closely overlaps early organogenesis and supports expression–regulatory comparisons. RNA and accessibility are different feature spaces; inspect sample stages rather than assuming all quarter-day points. |
| 34 | **scNMT-seq gastrulation multi-omics**, Argelaguet et al. 2019 | E4.5, E5.5, E6.5, **E7.5**; additional embryoid-body experiments in the parent record | [GSE121708](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121708), [author repository](https://github.com/rargelaguet/scnmt_gastrulation); RNA counts, methylation and accessibility products | Useful for germ-layer regulatory changes. Select in vivo embryo subseries **GSE121650 / GSE133725** for RNA; do not mix in vitro embryoid bodies into a normal embryo time course. |
| 35 | **ST-MAGIC integrated gastrulation multi-omic atlas**, 2026 | E6.5, E6.75, E7.0, E7.25, **E7.5** | [Paper/data availability](https://www.nature.com/articles/s41467-026-68291-w), [raw GSA CRA020056](https://ngdc.cncb.ac.cn/gsa/browse/CRA020056), [interactive portal](https://stmagic.miracle.ac.cn:29909/) | Paired RNA/ATAC with spatial inference using existing Geo-seq. **Raw deposit confirmed; other data are described as available on request**, so do not assume a public processed whole-atlas download. Reuses spatial references in entries 5–6. |
| 36 | **Early gastrulation / pluripotency atlas**, Mohammed et al. 2017 | E3.5, E4.5, E5.5, E6.5 | [GSE100597](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100597); QC-filtered compressed count table | Upstream cell-state reference and a TOME source dataset. Earlier than the target window. |
| 37 | **Pregastrulation pluripotency transition atlas**, Cheng et al. 2019 | Pregastrulation; TOME distributes E5.25, E5.5, E6.25 and E6.5 contributions | [GSE109071](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109071); expression tables and sample metadata; [TOME stage mapping](https://tome.gs.washington.edu/) | Useful for anterior/posterior epiblast and visceral endoderm precursors. Consult original sample labels for each download. |
| 38 | **Preimplantation expression programs**, Xue et al. 2013 | Oocyte through cleavage/morula stages | [GSE44183](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE44183); `mouse_expression_mat.txt.gz` | Early developmental baseline. This accession contains **human and mouse** data; use the explicitly mouse matrix. Not spatial and not close to E7.5. |
| 39 | **mdCLA — mouse developmental Cell and Lineage Atlas**, 2026 | 37 time points from 2-cell stage to birth; explicit **E7.5, E7.75, E8.0** count files | [Paper](https://www.nature.com/articles/s41556-026-01971-3), [processed OMIX014646](https://ngdc.cncb.ac.cn/omix/release/OMIX014646); stage-specific `.h5ad` counts, raw **CRA003171 / CRA039992** | Strong additional exact-stage match. Repository entries **OMIX014646-33 / -34 / -35** contain E7.5 (~597 MB), E7.75 (~178 MB), E8.0 (~1.47 GB). The deposit also includes later **E14.5/E18.0 kidney Xenium** files; these are separate spatial samples, not whole-embryo spatial coverage. |

## C. Organ-specific developmental single-cell atlases

These extend the anchor paper's early germ-layer and organ-primordium questions into more specialized tissues. For spatial versions of later gut or vascular development, see entries 14–15.

| # | Dataset | Stages / tissue | Download and available products | Relevance / caveat |
| --- | --- | --- | --- | --- |
| 40 | **Foregut endoderm–mesoderm signaling atlas**, Han et al. 2020 | E8.5, E9.0, E9.5; dissected foregut | [GSE136689](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE136689), [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7453027/); counts and annotations | Excellent follow-up to early endoderm/mesoderm interactions. Regional dissections provide coarse anatomy, not intact per-cell coordinates. |
| 41 | **Molecular architecture of the developing mouse brain**, La Manno et al. 2021 | Embryonic brain from gastrulation toward birth | [Paper](https://www.nature.com/articles/s41586-021-03775-x), [author download portal](http://mousebrain.org/development/downloads.html); processed expression/annotation objects | Large neural-development reference. The author portal timed out during this search; retain the paper's data-availability route if it remains unavailable. |
| 42 | **Molecular logic of cortical diversification**, Di Bella et al. 2021 | E10.5–P4; developing cerebral cortex | [GSE153164](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153164), [Single Cell Portal SCP1290](https://singlecell.broadinstitute.org/single_cell/study/SCP1290/molecular-logic-of-cellular-diversification-in-the-mammalian-cerebral-cortex); RNA and separate ATAC products | Later ectoderm/neural fate reference. Includes perturbation samples; select wild type for a baseline trajectory. |
| 43 | **Mouse Spinal Cord Atlas**, Delile et al. 2019 | E9.5–E13.5; developing spinal cord | [Author data and analysis repository](https://github.com/juliendelile/MouseSpinalCordAtlas), [E-MTAB-7320](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-7320); UMI counts and annotated cell metadata | Useful for neural-tube differentiation and dorsoventral identity. The paper's spatial interpretation does not make this a measured spatial expression matrix. |
| 44 | **Pancreatic endocrinogenesis**, Bastidas-Ponce et al. 2019 | E12.5, E13.5, E14.5, E15.5 | [GSE132188](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132188); 10x single-cell count products | Endoderm-derived organ development and differentiation. Ngn3-reporter sorting enriches endocrine progenitors, affecting apparent population proportions. |
| 45 | **Cross-platform kidney development atlas** | E14.5 kidney | [GSE104396](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104396), [study description](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA412523); Drop-seq, 10x Chromium, and Fluidigm C1 data | Mesoderm-derived organogenesis and platform comparison. Different technologies require separate normalization/technical interpretation. |

## D. Adult and mixed-age reference atlases

These are secondary choices for the present project: useful for mature cell identities, spatial-method benchmarking, and adult endpoints, but generally weaker references for E7.5–E8.0 progenitors.

| # | Dataset | Age / tissue | Modality and download | Practical value |
| --- | --- | --- | --- | --- |
| 46 | **Tabula Muris** | Adult; 20 organs/tissues | scRNA-seq; [GSE109774](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE109774), [official paper and Figshare download links](https://www.nature.com/articles/s41586-018-0590-4) | Broad mature-cell reference, with droplet and plate-based data. |
| 47 | **Tabula Muris Senis** | Multiple postnatal/adult/aged cohorts and tissues | scRNA-seq; [GSE149590](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149590), [project portal](https://tabula-muris-senis.sf.czbiohub.org/); `.h5ad`, `.loom`, and metadata | Convenient Python-ready organism-wide reference. Use age metadata; this is an aging study rather than an embryo atlas. |
| 48 | **Mouse Cell Atlas / MCA**, Han et al. 2018 | Mixed adult, neonatal, and selected fetal/placental material | Microwell-seq; [GSE108097](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108097); expression tables and sample archives | Broad tissue coverage with useful developmental subsets. The repository also includes human/mouse technical mixtures; filter samples by species and tissue. |
| 49 | **Allen Brain Cell Atlas / whole mouse brain** | Adult whole brain | sc/snRNA-seq and MERFISH; [official data-access documentation](https://alleninstitute.github.io/abc_atlas_access/), [MERFISH dataset description](https://alleninstitute.github.io/abc_atlas_access/descriptions/MERFISH-C57BL6J-638850.html) | Expression matrices, annotations, coordinates, and anatomical registration. Keep measured MERFISH panel expression separate from whole-transcriptome imputed releases. |
| 50 | **STARmap PLUS mouse CNS atlas** | Adult brain and spinal cord | Targeted spatial in situ sequencing; [Zenodo](https://doi.org/10.5281/zenodo.8327576), [SCP1830](https://singlecell.broadinstitute.org/single_cell/study/SCP1830), [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10709140/) | Large spatial benchmark: approximately 1.09 million cells with 1,022 measured genes. Useful for testing panel limitations and spatial cell typing. |
| 51 | **osmFISH somatosensory cortex** | Mouse brain cortex; mature tissue benchmark | Targeted spatial imaging; [author project/data](http://linnarssonlab.org/osmFISH/), [paper](https://www.nature.com/articles/s41592-018-0175-z) | Compact measured-expression benchmark. Processed data are public; the paper describes the very large raw image collection as available on request. |
| 52 | **Slide-seqV2 reference datasets** | Adult hippocampus and developing neocortex experiments; ages vary by puck | Spatial sequencing; [SCP815](https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2); DGE `.txt.gz`, bead-location `.csv`, BAMs | Useful cross-platform tests. The collection includes original Slide-seq controls as well as Slide-seqV2; identify the assay and age for each puck. |
| 53 | **Liver Cell Atlas** | Adult liver; steady-state and disease/diet cohorts | scRNA-seq, CITE-seq, and Visium; [official downloads](https://livercellatlas.org/download.php), [GSE192742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE192742) | Mature endoderm-derived tissue with spatial and single-cell references. Choose mouse steady-state data explicitly; the portal also contains human and disease data. |

## E. Additional databases, anatomical resources, and download routes

These are **collections or access routes**, not extra independent expression experiments to add to the 53 entries above.

| Resource | What it contributes | Download / access notes |
| --- | --- | --- |
| **NCBI GEO + SRA** | Primary archive for mouse expression studies, including many above | [GEO](https://www.ncbi.nlm.nih.gov/geo/); processed files under Supplementary Files, sample metadata under GSM records, sequencing reads via SRA. Search species, stage, tissue, and assay together. |
| **BioStudies / ArrayExpress** | Early mouse development studies deposited under E-MTAB accessions | [ArrayExpress collection](https://www.ebi.ac.uk/biostudies/arrayexpress); use study file tables and SDRF metadata. Especially relevant to the Pijuan-Sala, extended embryo, Scialdone, and Tyser datasets. |
| **Single Cell Expression Atlas** | Curated single-cell expression and experiment metadata | [Official download documentation](https://www.ebi.ac.uk/gxa/sc/download); per-experiment raw/normalized matrices, annotations, markers, and FTP downloads. Filter to *Mus musculus*. |
| **CZ CELLxGENE Discover** | Searchable single-cell and spatial collections | [Download documentation](https://cellxgene.cziscience.com/docs/03__Download%20Published%20Data); `.h5ad` downloads are convenient for this project. A standardized copy may be the same biological dataset already listed above. |
| **STOmicsDB / CNGBdb** | Spatial datasets including MOSTA and MPSTA | [STOmicsDB and download help](https://db.cngb.org/stomics/help?anchor=3); filter mouse/developmental tissue. Distinguish original `.h5ad` files from portal-reprocessed products. |
| **SpatialDB** | Curated spatial expression datasets across technologies | [Download table](https://spatialomics.org/SpatialDB/download.php); includes Geo-seq, Slide-seq, and other mouse resources, plus non-mouse experiments. Usually expression archives and spatially variable gene results. |
| **MouseGastrulationData** | Programmatic access to curated early embryo datasets | [Bioconductor package](https://bioconductor.org/packages/release/data/experiment/html/MouseGastrulationData.html), [dataset overview](https://bioconductor.org/packages/devel/data/experiment/vignettes/MouseGastrulationData/inst/doc/MouseGastrulationData.html); includes gastrulation RNA, perturbation data, E8.25 ATAC, and E8.5 seqFISH. An access wrapper, not a new atlas. |
| **MGI Gene Expression Database / GXD** | Mouse developmental expression evidence, anatomy, in situ images, and curated RNA-seq study metadata | [Official description and search routes](https://www.informatics.jax.org/mgihome/GXD/aboutGXD.shtml). Useful for checking candidate-gene locations; not uniformly a cell-by-gene matrix. |
| **Allen Developing Mouse Brain Atlas** | Spatial in situ expression and anatomical reference volumes across embryonic/postnatal brain development | [ISH portal](https://developingmouse.brain-map.org/), [download/API documentation](https://brain-map.org/support/documentation/allen-developing-mouse-brain-atlas-supplementary-data). Useful for spatial validation; ISH intensity volumes are not single-cell RNA counts. |
| **EMAGE / eMouseAtlas** | Historical spatial gene-expression annotations and embryo reference anatomy | [Resource paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC1347369/). Legacy infrastructure; current bulk-download availability was not confirmed. Treat as an anatomical/validation lead rather than a guaranteed ready-to-load transcriptomic dataset. |
| **Janelia Mouse Atlas / TARDIS** | Whole-embryo live imaging, cell tracks, divisions, fate maps, and morphogenesis | [Official download page](https://www.janelia.org/janelia-mouse-atlas). Closely related to gastrulation/early organogenesis, but **not a transcriptomic expression matrix**. Valuable if combining morphology and lineage with RNA atlases. |
| **GUDMAP legacy developmental resources** | Kidney and genitourinary development, including mouse expression/anatomy data | [Consortium resource and preservation notice](https://strandlab.net/GUDMAP/). ATLAS-D2K was sunsetted in July 2025; use surviving GEO, MGI, and CELLxGENE deposits rather than assuming the old portal still provides downloads. |

## Stage-focused selection guide

| Need | Most relevant entries | Important qualification |
| --- | --- | --- |
| Measured spatial data at all three exact stages | **1** | The closest match is the paper's own dataset. |
| Other measured spatial information at E7.5 | **3, 5, 6, 13** | Different anatomical coverage and spatial units; MPSTA includes maternal/extraembryonic tissue. |
| scRNA-seq at E7.5, E7.75, E8.0 | **19, 20, 21, 39** | Entries 19–21 share source data; mdCLA is an additional study with stage-specific files. |
| Independent early gastrulation time course | **23** | Use embryo metadata and distinguish collection age from inferred developmental time. |
| Early heart/cardiovascular gene selection | **29, 30, 31, 32** | Entry 31 explicitly samples E7.75; entry 32 is earlier. |
| Early endoderm and mesoderm–endoderm signaling | **1, 5, 6, 28, 40** | Entry 40 starts at E8.5, after the anchor window. |
| Experimental lineage information | **26, 27** | MELA includes E7.5/E8.0 samples; NextCell measures an E13.5 endpoint. |
| Measured spatial data immediately after the target window | **4, 8** | E8.5 is a close neighbor, not interchangeable with E8.0. |
| Later whole-embryo spatial trajectories | **2, 8, 9, 10, 11, 16, 17** | Confirm stage, embryo, section, and spatial resolution for each file. |
| Prenatal-to-birth expression reference | **24, 39** | Large resources; subset by developmental stage and tissue before loading everything. |
| Mature cell-type or spatial-method reference | **46–53** | Adult identities may not transfer cleanly to embryonic progenitors. |

## Download and comparison notes for the gene-panel project

1. **Prefer processed expression plus metadata first.** For Python, entries 1, 2, 9, 13, 21, 39, and 47 advertise `.h5ad` products. R/Seurat objects, H5TD, generic HDF5, MTX, and plain tables require their respective loaders or conversion.
2. **Retain biological replicate identity.** Cells, beads, and sections from the same embryo are not independent embryos. Do not split neighboring sections from one embryo across train/test sets and describe the result as cross-embryo validation.
3. **Keep measured and imputed genes separate.** This is especially important for seqFISH-based atlases and Allen MERFISH releases. A gene's presence in an imputed matrix does not establish that it was experimentally measured by that panel.
4. **Avoid duplicate biological data across portals.** TOME, the original/extended gastrulation atlases, MouseGastrulationData, and CELLxGENE can expose overlapping cells. MOSTA and Spateo also share download hosting for some files.
5. **Inspect expression units and gene identifiers.** Counts, log-normalized expression, FPKM, and imaging-derived counts are different inputs. Match stable gene IDs where possible and retain symbols separately; a shared gene symbol alone does not guarantee identical feature definitions.
6. **Check actual tissue coordinates.** Some datasets store these outside `obsm["spatial"]`, in per-cell metadata or separate files. Preserve source coordinates and units; reconstructed 3D coordinates need not be directly comparable across embryos.
7. **Use a staged download strategy.** Start with entry 1 plus the matched early-stage portion of entry 20 or 21. Add an independent spatial technology and later-stage data once the core loading and comparison workflow works. Large raw reads and image bundles are optional unless reprocessing is the goal.

The priorities above are project-specific recommendations based on stage, tissue, modality, and download convenience, rather than claims of a universal dataset quality ranking.
