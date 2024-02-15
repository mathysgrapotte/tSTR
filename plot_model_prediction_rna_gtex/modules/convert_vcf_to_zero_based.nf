process CONVERT_VCF_TO_ZERO_BASED {
    // VCF is 1-based, so we need to convert it to 0-based so that it can be intersected with bed files
    // That means we need to add the "chr" prefix to the chromosome name (first column)
    // We need to duplicate the second column (position) and substract 1 (this will be the new second column)
    // for instance, 1 1000 will become chr1 999 1000
    // The rest of the VCF should be kept as is, we need to ignore the header lines (starting with #) but still copy them to the output header

    input:
    path(vcf_file)

    output:
    path('zero_based_vcf.vcf'), emit: zero_based_vcf

    script:
    """
    awk 'BEGIN{{OFS="\\t"}}{{if(\$1 ~ /^#/){{print \$0}}else{{print "chr"\$1,\$2-1,\$2,\$0}}}}' $vcf_file > zero_based_vcf.vcf
    """

    stub:
    // do the same thing, but only on the first thousand lines of the vcf file
    """
    head -n 1000 $vcf_file | awk 'BEGIN{{OFS="\\t"}}{{if(\$1 ~ /^#/){{print \$0}}else{{print "chr"\$1,\$2-1,\$2,\$0}}}}' > zero_based_vcf.vcf
    """
}