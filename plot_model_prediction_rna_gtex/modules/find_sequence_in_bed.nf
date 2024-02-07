process FIND_SEQUENCE_IN_BED {
    // this process will use a regex to output the line in a bed file that countains the sequence name of interest

    label "process_low"

    input:
    path(bed_file)
    val(sequence_name)

    output:
    path("filtered_bed.bed"), emit: filtered_bed

    script:
    """
    grep -E "$sequence_name" $bed_file > filtered_bed.bed
    """
}