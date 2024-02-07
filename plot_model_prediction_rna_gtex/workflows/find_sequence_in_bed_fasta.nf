include { FIND_SEQUENCE_IN_BED} from '../modules/find_sequence_in_bed.nf'

workflow FIND_SEQUENCE_IN_BED_FASTA {

    take:
    bed_file
    sequence_name
    
    main:

    FIND_SEQUENCE_IN_BED(bed_file, sequence_name)
    filtered_bed = FIND_SEQUENCE_IN_BED.out.filtered_bed

    emit:
    filtered_bed

}