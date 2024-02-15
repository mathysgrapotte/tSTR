process FIND_GENE_IN_GCT_GTEX {

    // this process takes as input a gct file and a gene name and outputs a filtered gct file
    // the gct contains two header lines that are not useful for the filtering 
    // those two header lines are followed by one header line that has the name, description and sample ids, those should be kept in the output
    // the header lines do not necessarly have the '#' character at the beginning of the line, but they are always the first lines of the file
    // the rest of the lines are the composed of the gene name, the description and the expression values for each sample
    // this process should filter the gct file to keep only the lines that have the gene name in the gene name column using a minimal bash script

    input:
    path(gct_file)
    val(gene_name)

    output:
    path("filtered_gct.gct"), emit: filtered_gct

    script:
    """
    # remove the first two lines and keep the third line
    head -n 3 $gct_file | tail -n 1 > filtered_gct.gct
    # add the lines that have the gene name in the gene name column
    grep -E "^$gene_name" $gct_file >> filtered_gct.gct
    """



}