// this process filters the input VCF by a region in a single line bed
process FILTER_VCF_SEQUENCE_IN_BED {

    // this process uses bcftools to filter the input VCF by the coordinates in the BED file
    // please note that the BED file must contain exactly one sequence
    // We first remove the chr prefix from the first column of the BED file
    // Then, we use bcftools to filter the VCF by the coordinates in the BED file

    container 'blcdsdockerregistry/bcftools@sha256:9f2d6573bb5d200c9e6e478568e812f100c5448c6e90e8a22d229acdeaba2229'

    input: 
    path(vcf)
    path(vcf_index)
    path(bed)

    output:
    path("filtered_vcf.vcf"), emit: filtered_vcf

    script:
    """
    # check that the bed has only one entry
    if [ \$(wc -l ${bed} | awk '{print \$1}') -ne 1 ]; then
        echo "The BED file must contain exactly one region"
        exit 1
    fi

    # remove the chr prefix from the first column of the BED file
    sed 's/chr//g' ${bed} > temp.bed

    # filter the VCF by the coordinates in the BED file
    bcftools view -R temp.bed ${vcf} -o filtered_vcf.vcf

    """
}