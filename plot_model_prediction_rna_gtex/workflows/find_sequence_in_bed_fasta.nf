include { FIND_SEQUENCE_IN_BED} from '../modules/find_sequence_in_bed.nf'
include { FIND_SEQUENCE_IN_FASTA} from '../modules/find_sequence_in_fasta.nf'

workflow FIND_SEQUENCE_IN_BED_FASTA {

    take:
    fasta_file
    bed_file
    sequence_name

    main:

    FIND_SEQUENCE_IN_BED(bed_file, sequence_name)
    filtered_bed = FIND_SEQUENCE_IN_BED.out.filtered_bed


    FIND_SEQUENCE_IN_FASTA(fasta_file, sequence_name)
    filtered_fasta = FIND_SEQUENCE_IN_FASTA.out.filtered_fasta

    emit:
    filtered_bed
    filtered_fasta


}