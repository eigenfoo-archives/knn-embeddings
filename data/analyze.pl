# Analyze results

$ARGC = @ARGV;
if ($ARGC != 2) {
    print STDERR "Usage: analyze.pl PREDICTIONS ACTUAL\n";
    print STDERR "   PREDICTION: list of system's predictions\n";
    print STDERR "   ACTUAL: list of the actual labels\n";
    exit;
}

$results = $ARGV[0];
$actual = $ARGV[1];

# Process results file.

print STDERR "Processing answer file...\n";

if (!(open(ANSWER, "<$actual"))) {
    print STDERR "Error: could not open $actual for input.\n";
    exit;
}

# Load the category information

# Stores size of each category
%cat_size = ();
# Set to true for seen categories
%have_seen = ();
# Stores the actual category of every document.
%actual_cat = ();

while ($line = <ANSWER>) {
    if (!($line =~ /([^ ]*) (.*)/)) {
        print STDERR "Error: Line is actual file not in expected format!\n";
        exit;
    }
    $number = $1;
    $actual_cat{$number} = $2;
    $have_seen{$2} = 1;
    $cat_size{$2} += 1;
}

# Stores list of categories
@categories = keys %have_seen;
$num_categories = @categories;

print "Found $num_categories categories:";
foreach $cat (@categories) {
    print " $cat";
}
print "\n";

# Process results file.

print STDERR "Processing prediction file...\n";

if (!(open(RESULTS, "<$results"))) {
    print STDERR "Error: could not open $results for input.\n";
    exit;
}

# Stores the predicted category of every document.
%predicted_cat = ();
# First dimension of contingency table (row) is system's prediction,
# second dimension (column) is actual category.
%contingency = ();

while ($line = <RESULTS>) {
    if (!($line =~ /([^ ]*) (.*)/)) {
	print STDERR "Error: Line in results file not in expected format!\n";
	exit;
    }
    $number = $1;
    $predicted_cat{$number} = $2;

    $contingency{$predicted_cat{$number}}{$actual_cat{$number}} += 1;
}

# Determine overall accuracy.
$correct = $incorrect = 0;
foreach $cat (@categories) {
    foreach $cat2 (@categories) {
	if ($cat eq $cat2) {
	    $correct += $contingency{$cat}{$cat2};
	}
	else {
	    $incorrect += $contingency{$cat}{$cat2};
	}
    }
}
if ($correct + $incorrect > 0) {
    $ratio = $correct / ($correct + $incorrect);
}
else {
    $ratio = "UNDEFINED";
}
print "\n$correct CORRECT, $incorrect INCORRECT, RATIO = $ratio.\n\n";

# Display contingency table, calculate precisions, recalls, and F_1s.

# Header row.
print "CONTINGENCY TABLE:\n        ";
foreach $cat (@categories) {
    if (length($cat) > 7) {
	$header = substr($cat, 0, 7);
    }
    else {
	$header = $cat;
    }
    printf ("%-8s", $header);
}
print "PREC\n";

# Category rows.
%precision = ();
foreach $cat (@categories) {
    if (length($cat) > 7) {
	$header = substr($cat, 0, 7);
    }
    else {
	$header = $cat;
    }
    printf ("%-8s", $header);

    $correct = $incorrect = 0;
    foreach $cat2 (@categories) {
	if ($contingency{$cat}{$cat2} eq "") {
	    $contingency{$cat}{$cat2} = 0;
	}
	printf ("%-8d", $contingency{$cat}{$cat2});
	if ($cat eq $cat2) {
	    $correct += $contingency{$cat}{$cat2};
	}
	else {
	    $incorrect += $contingency{$cat}{$cat2};
	}
    }
    if ($correct + $incorrect > 0) {
	$prec = $correct / ($correct + $incorrect);
    }
    else {
	$prec = 0;
    }
    $precision{$cat} = $prec;
    printf("%.2f\n", $prec);
}

# Recall row.
%recall = ();
print "RECALL  ";
foreach $cat (@categories) {
    $correct = $incorrect = 0;
    foreach $cat2 (@categories) {
	# Now $cat is the column and $cat2 is the row.
	if ($cat eq $cat2) {
	    $correct += $contingency{$cat2}{$cat};
	}
	else {
	    $incorrect += $contingency{$cat2}{$cat};
	}	
    }
    if ($correct + $incorrect > 0) {
	$rec = $correct / ($correct + $incorrect);
    }
    else {
	$rec = 0;
    }
    $recall{$cat} = $rec;
    printf("%-8.2f", $rec);    
}
print "\n\n";

# F_1 values.
%f1 = ();
foreach $cat (@categories) {
    if ($precision{$cat} + $recall{$cat} > 0) {
	$f1{$cat} = (2 * $precision{$cat} * $recall{$cat}) /
	    ($precision{$cat} + $recall{$cat});
    }
    else {
	$f1{$cat} = 0;
    }
    print "F_1($cat) = $f1{$cat}\n";
}
print "\n";
