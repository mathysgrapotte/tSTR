include { FIND_SEQUENCE_IN_BED_FASTA} from './workflows/find_sequence_in_bed_fasta.nf'

workflow {

    // Define the input parameters
    input_fasta = Channel.fromPath(params.fasta)
    input_bed = Channel.fromPath(params.bed)
    input_sequence_name = Channel.of(params.sequence_name)

    FIND_SEQUENCE_IN_BED_FASTA(input_fasta, input_bed, input_sequence_name)
    filtered_bed = FIND_SEQUENCE_IN_BED_FASTA.out.filtered_bed
    filtered_fasta = FIND_SEQUENCE_IN_BED_FASTA.out.filtered_fasta
}

