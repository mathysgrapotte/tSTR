include {FIND_GENE_IN_GCT_GTEX} from '../modules/find_gene_in_gct_gtex.nf'

workflow FIND_GENE_IN_GCT {
    take:
    gct_file
    gene_name

    main:
    FIND_GENE_IN_GCT_GTEX(gct_file, gene_name)
    filtered_gct = FIND_GENE_IN_GCT_GTEX.out.filtered_gct

    emit:
    filtered_gct
}