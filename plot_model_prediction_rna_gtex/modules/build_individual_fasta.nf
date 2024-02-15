process BUILD_INDIVIDUAL_FASTA {

    container "alessiovignoli3/model-check:generate_fasta"

    input:
    path(fasta_file)
    path(bed_file)
    path(vcf_file)

    output:
    path("individual_fasta.fa"), emit: individual_fasta

    script:
    """
    launch_individual_fasta.py -f ${fasta_file} -b ${bed_file} -v ${vcf_file} -o individual_fasta.fa
    """
}