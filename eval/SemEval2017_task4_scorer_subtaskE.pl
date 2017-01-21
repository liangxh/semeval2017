#!/usr/bin/perl
#
#  Author: Sara Rosenthal, Preslav Nakov
#  
#  Description: Scores SemEval-2016 task 4, subtask E
#               Calculates Earth Mover's Distance
#
#  Last modified: December 29, 2016
#
# Use:
# (a) outside of CodaLab
#     perl SemEval2017_task4_test_scorer_subtaskE.pl <GOLD_FILE> <INPUT_FILE>
# (b) with CodaLab, i.e., $codalab=1 (certain formatting is expected)
#     perl SemEval2017_task4_test_scorer_subtaskE.pl <INPUT_FILE> <OUTPUT_DIR>
#


use warnings;
use strict;
use utf8;
binmode (STDIN,  ":utf8");
binmode (STDOUT, ":utf8");

my $GOLD_FILE          =  $ARGV[0];
my $INPUT_FILE         =  $ARGV[1];
my $OUTPUT_FILE        =  $INPUT_FILE . '.scored';

my $codalab = 0; # set to 1 if the script is being used in CodaLab

########################
###   MAIN PROGRAM   ###
########################

my %trueStats = ();
my %proposedStats = ();

### 1. Read the files and get the statistics
if ($codalab) {
	my $INPUT_DIR = $ARGV[0];
	print STDERR "Loading input from dir: $INPUT_DIR\n";
 
	opendir(DIR, "$INPUT_DIR/res/") or die $!;

	while (my $file = readdir(DIR)) {

	    # Use a regular expression to ignore files beginning with a period
    	    next if ($file =~ m/^(\.|_)/);
	    $INPUT_FILE = "$INPUT_DIR/res/$file";
	    last;
	}
	closedir(DIR);
	$GOLD_FILE   = "$INPUT_DIR/ref/SemEval2016_task4_subtaskE_test_gold.txt";
	$OUTPUT_FILE = $ARGV[1] . "/scores.txt";
}

print STDERR "Found input file: $INPUT_FILE\n";
open INPUT, $INPUT_FILE or die;

print STDERR "Loading ref data $GOLD_FILE\n";
open GOLD,  $GOLD_FILE or die;

print STDERR "Loading the file...";

my $totalExamples = 0;

for (<INPUT>) {
	s/^[ \t]+//;
	s/[ \t\n\r]+$//;

	### 1.1. Check the input file format
	#michael jordan   .3  .6  .0  .05 .05
	die "Wrong format in input: ", $_ if (!/^([^\t]+)\t([\d\.]+)\t([\d\.]+)\t([\d\.]+)\t([\d\.]+)\t([\d\.]+)/);
	my $topic = $1;
	$trueStats{$topic}{'-2'} = $2;
	$trueStats{$topic}{'-1'} = $3;
	$trueStats{$topic}{'0'} = $4;
	$trueStats{$topic}{'1'} = $5;
	$trueStats{$topic}{'2'} = $6;

	my $sum = 0.0;
	foreach my $class (keys $trueStats{$topic}) {
	    my $p = $trueStats{$topic}{$class};
	    die "Number not in range $p" if ($p < -0.0001 || $p > 1.0001);  
	    $sum += $trueStats{$topic}{$class};
	}
	die "Probabilities do not sum to 1, ($sum) topic ", $topic if (abs($sum - 1.0) > .001 && abs($sum) > .001);

	### 1.2	. Check the prediction file format (same as above)
	$_ = <GOLD>;
	die "Wrong format in gold: ", $_ if (!/^([^\t]+)\t([\d\.]+)\t([\d\.]+)\t([\d\.]+)\t([\d\.]+)\t([\d\.]+)/);
	my $proposedTopic = $1;
	$proposedStats{$proposedTopic}{'-2'} = $2;
	$proposedStats{$proposedTopic}{'-1'} = $3;
	$proposedStats{$proposedTopic}{'0'} = $4;
	$proposedStats{$proposedTopic}{'1'} = $5;
	$proposedStats{$proposedTopic}{'2'} = $6;

	$sum = 0.0;
	foreach my $class (keys $proposedStats{$proposedTopic}) {
	    my $p = $proposedStats{$topic}{$class};
	    die "Number not in range $p" if ($p < -0.0001 || $p > 1.0001);  
	    $sum += $proposedStats{$proposedTopic}{$class};
	}
	die "Probabilities do not sum to 1, ($sum) topic ", $proposedTopic if (abs($sum - 1.0) > .01);

	die "Topic mismatch: gold:'$topic' <> proposed:'$proposedTopic'" if ($topic ne $proposedTopic);
}

while (<GOLD>) {
	die "Missing answer for the following tweet: '$_'\n";
}
print STDERR "DONE\n";

close(INPUT) or die;
close(GOLD) or die;

print STDERR "Calculating the scores...\n";

my @labels = ('-2', '-1', '0', '1', '2');

### 2. Calculate KL divergence for each topic and average
open OUTPUT, '>:encoding(UTF-8)', $OUTPUT_FILE or die;

my $overall = 0.0;
my $numTopics = 0;
foreach my $topic (sort keys %trueStats) {
    my $emd = 0.0;
    for my $ind1 (0 .. $#labels - 1) {
    	my ($sumTrue, $sumProposed) = (0.0, 0.0);
		for my $ind2 (0 .. $ind1) {
		    my $class = $labels[$ind2];
		    $sumTrue += $trueStats{$topic}{$class};
		    $sumProposed += $proposedStats{$topic}{$class};
		}
		$emd += abs($sumTrue - $sumProposed);
    }
    $overall += $emd;
    $numTopics ++;
    printf OUTPUT "\t%18s: EMD=%0.3f\n", $topic, $emd if (!$codalab);
}
$overall /= $numTopics;

if ($codalab) {
	printf OUTPUT "EMD: %0.3f\n", $overall;
} else {
	printf OUTPUT "\tOVERALL EMD : %0.3f\n", $overall;
	print "$INPUT_FILE\t";
}
printf "%0.3f\t", $overall;

print "\n";
close(OUTPUT) or die;
