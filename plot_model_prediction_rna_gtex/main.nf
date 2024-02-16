include { FIND_SEQUENCE_IN_BED_FASTA} from './workflows/find_sequence_in_bed_fasta.nf'
include { INTERSECT_VCF_WITH_ONE_LINE_BED } from './workflows/intersect_vcf_with_one_line_bed.nf'
include { BUILD_FASTA} from './workflows/build_fasta.nf'
include { PREDICT} from './workflows/predict.nf'
include { FIND_GENE_IN_GCT } from './workflows/find_gene_in_gct.nf'
include { MERGE_TO_MAKE_PLOT_DATA } from './workflows/merge_to_make_plot_data.nf'
include { MAKE_PLOTS } from './workflows/make_plots.nf'

workflow {

    // Define the input parameters
    input_vcf = Channel.fromPath(params.vcf)
    input_vcf_index = Channel.fromPath(params.vcf_index)
    input_fasta = Channel.fromPath(params.fasta)
    input_bed = Channel.fromPath(params.bed)
    input_gct = Channel.fromPath(params.gct)
    input_metadata = Channel.fromPath(params.meta)
    model_architecture = Channel.fromPath(params.ma)
    model_weights = Channel.fromPath(params.mw)
    input_gene_name = Channel.of(params.gene_name)
    input_sequence_name = Channel.of(params.sequence_name)

    FIND_GENE_IN_GCT(input_gct, input_gene_name)
    filtered_gct = FIND_GENE_IN_GCT.out.filtered_gct

    FIND_SEQUENCE_IN_BED_FASTA(input_fasta, input_bed, input_sequence_name)
    filtered_bed = FIND_SEQUENCE_IN_BED_FASTA.out.filtered_bed
    filtered_fasta = FIND_SEQUENCE_IN_BED_FASTA.out.filtered_fasta

    INTERSECT_VCF_WITH_ONE_LINE_BED(input_vcf, input_vcf_index, filtered_bed)
    filtered_vcf = INTERSECT_VCF_WITH_ONE_LINE_BED.out.filtered_vcf

    BUILD_FASTA(filtered_fasta, filtered_bed, filtered_vcf)
    individual_fasta = BUILD_FASTA.out.individual_fasta

    PREDICT(model_weights, model_architecture, individual_fasta)
    predictions = PREDICT.out.predictions

    MERGE_TO_MAKE_PLOT_DATA(predictions, input_metadata, filtered_gct)
    plot_data = MERGE_TO_MAKE_PLOT_DATA.out.plot_data

    MAKE_PLOTS(plot_data)
    scatter_plots = MAKE_PLOTS.out.scatter_plots


}

