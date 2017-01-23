#!/usr/bin/perl
#
#  Author: Sara Rosenthal, Preslav Nakov
#  
#  Description: Scores SemEval-2017 task 4, subtask D
#               using Kullback-Leibler Divergence and Pearson Divergence
#
#  Last modified: January 3, 2017
#
# Use:
# (a) outside of CodaLab
#     perl SemEval2017_task4_test_scorer_subtaskD.pl <GOLD_FILE> <INPUT_FILE>
# (b) with CodaLab, i.e., $codalab=1 (certain formatting is expected)
#     perl SemEval2017_task4_test_scorer_subtaskD.pl <INPUT_FILE> <OUTPUT_DIR>
#

use warnings;
use strict;
use utf8;
binmode (STDIN,  ":utf8");
binmode (STDOUT, ":utf8");

my $GOLD_FILE   =  $ARGV[0];
my $INPUT_FILE  =  $ARGV[1];
my $OUTPUT_FILE =  $INPUT_FILE . '.scored';

my $codalab = 0; # set to 1 if the script is being used in CodaLab


########################
###   MAIN PROGRAM   ###
########################

my %trueStatsSmoothed = ();
my %proposedStatsSmoothed = ();
my %trueStatsOrig = ();
my %proposedStatsOrig = ();
my %topicCounts = ();


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
	$GOLD_FILE   = "$INPUT_DIR/ref/twitter-2016test-D-English.txt";
	$OUTPUT_FILE = $ARGV[1] . "/scores.txt";
}

print STDERR "Found input file: $INPUT_FILE\n";
open INPUT, $INPUT_FILE or die;

print STDERR "Loading ref data $GOLD_FILE\n";
open GOLD,  $GOLD_FILE or die;

print STDERR "Loading the file...";

for (<GOLD>) {
	s/^[ \t]+//;
	s/[ \n\r]+$//;

	### 1.1. Check the input file format
	#michael jordan   .35  .65	10
	die "Wrong format in gold: ", $_ if (!/^([^\t]+)\t([\d\.]+)\t([\d\.]+)\t(\d+)/);
	my ($topic, $pos, $neg, $count) = ($1, $2, $3, $4);

	die "Probability not in range $pos" if ($pos < -0.0001 || $pos > 1.0001);  
	die "Probability not in range $neg" if ($neg < -0.0001 || $neg > 1.0001);  

	# check pos + neg sum to 1.0, but allow both to be 0 for format checking purposes 
	die "Probabilities do not sum to 1: ", ($pos + $neg) if (abs($pos + $neg - 1) > .0001 && abs($pos + $neg) > .001);

	### 1.2	. Check the prediction file format (same as input)
	$_ = <INPUT>;
	die "Wrong format in input: ", $_ if (!/^([^\t]+)\t([\d\.]+)\t([\d\.]+)/);
	my ($proposedTopic, $proposedPos, $proposedNeg) = ($1, $2, $3);
	die "Probability not in range $proposedPos" if ($proposedPos < -0.0001 || $proposedPos > 1.0001);  
	die "Probability not in range $proposedNeg" if ($proposedNeg < -0.0001 || $proposedNeg > 1.0001);  
	die "Probabilities do not sum to 1" if abs(($proposedPos + $proposedNeg - 1.0) > .001);

	die "Topic mismatch: gold:'$topic' <> proposed:'$proposedTopic'" if ($topic ne $proposedTopic);

	### 1.3. Update the statistics
	$trueStatsSmoothed{$topic}{'positive'} = $trueStatsOrig{$topic}{'positive'} = $pos;
	$trueStatsSmoothed{$topic}{'negative'} = $trueStatsOrig{$topic}{'negative'} = $neg;
	$proposedStatsSmoothed{$topic}{'positive'} = $proposedStatsOrig{$topic}{'positive'} = $proposedPos;
	$proposedStatsSmoothed{$topic}{'negative'} = $proposedStatsOrig{$topic}{'negative'} = $proposedNeg;
	$topicCounts{$topic} = $count;

}

while (<GOLD>) {
	die "Missing answer for the following tweet: '$_'\n";
}
print STDERR "DONE\n";

close(INPUT) or die;
close(GOLD) or die;

print STDERR "Calculating the scores...\n";

### 2. Initialize zero counts, compute probabilities and smooth
foreach my $topic (keys %trueStatsSmoothed) {
	my $epsilon = 1 / (2.0 * $topicCounts{$topic});
    foreach my $class (keys $trueStatsSmoothed{$topic}) {		
		# Smooth the probabilities by epsilon
		$trueStatsSmoothed{$topic}{$class} = ($trueStatsSmoothed{$topic}{$class} + $epsilon) / (1.0 + $epsilon * (scalar keys $trueStatsSmoothed{$topic}));
		$proposedStatsSmoothed{$topic}{$class} = ($proposedStatsSmoothed{$topic}{$class} + $epsilon) / (1.0 + $epsilon * (scalar keys $trueStatsSmoothed{$topic}));
    }
}

### 3. Calculate KL divergence for each topic and average
open OUTPUT, '>:encoding(UTF-8)', $OUTPUT_FILE or die;

my $KLD = 0.0;
my $numTopics = 0;
my $AE = 0.0;
my $RAE = 0.0;
foreach my $topic (sort keys %trueStatsSmoothed) {
    my $kl = 0.0;
	my $absDiff = 0.0;
	my $relAbsDiff = 0.0;
    foreach my $class (keys $trueStatsSmoothed{$topic}) {
		$kl += $trueStatsSmoothed{$topic}{$class} *
	    	log($trueStatsSmoothed{$topic}{$class}/$proposedStatsSmoothed{$topic}{$class});
	    $absDiff += abs($proposedStatsOrig{$topic}{$class} - $trueStatsOrig{$topic}{$class});
	    $relAbsDiff += 1.0 * abs($proposedStatsSmoothed{$topic}{$class} - $trueStatsSmoothed{$topic}{$class}) / $trueStatsSmoothed{$topic}{$class};
    }
    $KLD += $kl;
    my $curAE  = $absDiff / (scalar keys $trueStatsSmoothed{$topic});
    my $curRAE = $relAbsDiff / (scalar keys $trueStatsSmoothed{$topic});
    $AE += $curAE;
    $RAE += $curRAE;
    $numTopics++;
    printf OUTPUT "\t%18s: KL=%0.3f, AE=%0.3f, RAE=%0.3f\n", $topic, $kl, $curAE, $curRAE;
}
$KLD /= $numTopics;
$AE  /= $numTopics;
$RAE /= $numTopics;

if ($codalab) {
	printf OUTPUT "KLD: %0.3f\nAE: %0.3f\nRAE: %0.3f\n", $KLD, $AE, $RAE;
} else {
	printf OUTPUT "\tOVERALL : KLD=%0.3f, AE=%0.3f, RAE=%0.3f\n", $KLD, $AE, $RAE;
	print "$INPUT_FILE\t";
	printf "%0.3f\t%0.3f\t%0.3f\n", $KLD, $AE, $RAE;
}
	
printf STDERR"KLD: %0.3f\nAE: %0.3f\nRAE: %0.3f\n", $KLD, $AE, $RAE;
close(OUTPUT) or die;
