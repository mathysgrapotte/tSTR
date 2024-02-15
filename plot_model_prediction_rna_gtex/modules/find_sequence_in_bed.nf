process FIND_SEQUENCE_IN_BED {
    // this process will use a regex to output the line in a bed file that countains the sequence name of interest
    // this process also outputs the chromosome of the sequence name of interest, this can be found from the first column of the filtered bed file
    // the chromosome is in the format 'chrn' where n is the number of the chromosome, 'chr' will be filtered out
    // the chromosome will be outputted to a bash variable called 'chromosome'

    label "process_low"

    input:
    path(bed_file)
    val(sequence_name)

    output:
    path("filtered_bed.bed"), emit: filtered_bed
 

    script:
    """
    grep -w "$sequence_name" $bed_file > filtered_bed.bed

    """
}