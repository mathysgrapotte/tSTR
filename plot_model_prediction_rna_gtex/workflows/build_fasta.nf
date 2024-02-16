include {BUILD_INDIVIDUAL_FASTA} from '../modules/build_individual_fasta.nf'

workflow BUILD_FASTA {
    take:
    fasta_file
    bed_file
    vcf_file

    main:
    BUILD_INDIVIDUAL_FASTA(fasta_file, bed_file, vcf_file)
    individual_fasta = BUILD_INDIVIDUAL_FASTA.out.individual_fasta

    emit:
    individual_fasta
}