#!/usr/bin/perl
#
#  Author: Veselin Stoyanov
#  
#  Description: Scores SemEval-2016 task 4, subtask B
#               Calculates macro-average R
#
#  Last modified: Feb. 24, 2016
#
#

use warnings;
use strict;
use utf8;
binmode (STDIN,  ":utf8");
binmode (STDOUT, ":utf8");

my $GOLD_FILE          =  $ARGV[0];
my $INPUT_FILE         =  $ARGV[1];
my $OUTPUT_FILE        =  $INPUT_FILE . '.scored';


########################
###   MAIN PROGRAM   ###
########################

my %stats = ();

### 1. Read the files and get the statsitics
open INPUT, '<:encoding(UTF-8)', $INPUT_FILE or die;
open GOLD,  '<:encoding(UTF-8)', $GOLD_FILE or die;

for (; <GOLD>; ) {
	s/^[ \t]+//;
	s/[ \t\n\r]+$//;

	### 1.1. Check the input file format
	#1234	michael jackson positive
	die "Wrong file format!" if (!/^(\d+)\t[^\t]+\t(UNKNOWN|positive|negative)/);
	my ($pid, $trueLabel) = ($1, $2);

	### 1.2	. Check the prediction file format
	#14114531	michael jackson positive
	$_ = <INPUT>;
	die "Wrong file format!" if (!/^(\d+)\t[^\t]+\t(positive|negative)/);
	my ($tid, $proposedLabel) = ($1, $2);

        die "Ids mismatch!" if ($pid ne $tid);
	### 1.3. Update the statistics
	$stats{$proposedLabel}{$trueLabel}++;
}

close(INPUT) or die;
close(GOLD) or die;

#print "Format OK\n";

### 2. Initialize zero counts
foreach my $class1 ('positive', 'negative') {
    foreach my $class2 ('positive', 'negative') {
	$stats{$class1}{$class2} = 0 if (!defined($stats{$class1}{$class2}))
    }
}


### 3. Calculate the scores
print "$INPUT_FILE\t";
open OUTPUT, '>:encoding(UTF-8)', $OUTPUT_FILE or die;

my $overall = 0.0;
foreach my $class ('positive', 'negative') {
    my $denomP = (($stats{$class}{'positive'} + $stats{$class}{'negative'}) > 0) ? ($stats{$class}{'positive'} + $stats{$class}{'negative'}) : 1;
    my $P = $stats{$class}{$class} / $denomP;

    my $denomR = ($stats{'positive'}{$class} + $stats{'negative'}{$class}) > 0 ? ($stats{'positive'}{$class} + $stats{'negative'}{$class}) : 1;
    my $R = $stats{$class}{$class} / $denomR;
		
    my $denom = ($P+$R > 0) ? ($P+$R) : 1;
    my $F1 = 2*$P*$R / $denom;

    printf OUTPUT "\t%8s: P=%0.2f, R=%0.2f, F1=%0.2f\n", $class, $P, $R, $F1;
    $overall += $R;
}
$overall /= 2.0;
printf OUTPUT "\tOVERALL SCORE : %0.3f\n", $overall;
printf "%0.3f\t", $overall;

print "\n";
close(OUTPUT) or die;
