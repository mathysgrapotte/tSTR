include {INTERSECT_BED_VCF} from '../modules/intersect_bed_vcf.nf'

workflow INTERSECT {
    take:
    bed_file
    vcf_file

    main:
    INTERSECT_BED_VCF(bed_file, vcf_file)
    intersect_vcf = INTERSECT_BED_VCF.out.intersect_vcf

    emit:
    intersect_vcf

}