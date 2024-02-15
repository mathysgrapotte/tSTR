process INTERSECT_BED {
    // this process aims at running intersectBed on a VCF file and a BED file

    container 'biocontainers/bedtools@sha256:c042e405f356bb44cc0d7a87b4528d793afb581f0961db1d6da6e0a7e1fd3467'

    input:
    path(bed_file_1)
    path(bed_file_2)
    

    output:
    path("intersect_bed.bed"), emit: intersect_bed

    script:
    """
    intersectBed -a ${bed_file_1} -b ${bed_file-2}  -wa -wb > intersect_bed.bed
    """
}