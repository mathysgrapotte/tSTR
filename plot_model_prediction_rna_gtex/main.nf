include { FIND_SEQUENCE_IN_BED_FASTA} from './workflows/find_sequence_in_bed_fasta.nf'

workflow {

    input_bed = Channel.fromPath(params.bed)
    input_sequence_name = Channel.of(params.sequence_name)

    FIND_SEQUENCE_IN_BED_FASTA(input_bed, input_sequence_name)
    filtered_bed = FIND_SEQUENCE_IN_BED_FASTA.out.filtered_bed
}

