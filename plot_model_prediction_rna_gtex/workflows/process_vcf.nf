include {CONVERT_VCF_TO_ZERO_BASED} from '../modules/convert_vcf_to_zero_based.nf'
include {FILTER_VCF_BY_CHROMOSOME} from '../modules/filter_vcf_by_chromosome.nf'

workflow PROCESS_VCF {

    take:
    vcf_file
    chromosome

    main:
    FILTER_VCF_BY_CHROMOSOME(vcf_file, chromosome)
    filtered_vcf = FILTER_VCF_BY_CHROMOSOME.out.filtered_vcf

    CONVERT_VCF_TO_ZERO_BASED(filtered_vcf)
    zero_based_vcf = CONVERT_VCF_TO_ZERO_BASED.out.zero_based_vcf

    emit:
    zero_based_vcf
}