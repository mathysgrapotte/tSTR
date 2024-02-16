include { FILTER_VCF_SEQUENCE_IN_BED} from '../modules/filter_vcf_sequence_in_bed.nf'

workflow INTERSECT_VCF_WITH_ONE_LINE_BED {
    take:
    vcf_file
    vcf_index
    bed_file

    main:
    FILTER_VCF_SEQUENCE_IN_BED(vcf_file, vcf_index, bed_file)
    filtered_vcf = FILTER_VCF_SEQUENCE_IN_BED.out.filtered_vcf

    emit:
    filtered_vcf
}