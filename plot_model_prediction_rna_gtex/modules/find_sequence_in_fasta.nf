process FIND_SEQUENCE_IN_FASTA {
    // this process collects both the header and the sequence from a FASTA file when the header is matching a sequence name
    // each sequence in the fasta is separated by a header starting with '>'

    input:
    path(fasta_file)
    val(sequence_name)
    
    output:
    path("filtered_fasta.fa"), emit: filtered_fasta

    script:
    """
    grep -A 1 -w '$sequence_name' $fasta_file > filtered_fasta.fa
    """
}